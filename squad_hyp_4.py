import geoopt
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from geoopt import PoincareBall
#from geoopt.PoincareBall import mobius_add, mobius_matvec, expmap0, logmap0
#from geoopt.manifolds.poincare.math import mobius_add, mobius_matvec, expmap0, logmap0

#Load and Preprocess SQuAD############################################################
from transformers import BertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

from evaluate import load
metric = load("squad")
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.optim import Adam
from evaluate import load as load_metric


squad_metric = load_metric("squad")
bleu_metric = load_metric("bleu")



# Load tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_set = load_dataset('squad', split='train[:1%]')
val_set = load_dataset('squad', split='validation[:1%]')
max_len = 128
num_layers=3

# Filter examples with no answers
train_set = train_set.filter(lambda x: x['answers']['text'] and x['answers']['answer_start'])
val_set = val_set.filter(lambda x: x['answers']['text'] and x['answers']['answer_start'])


# Preprocessing
def preprocess(example):
    enc = tokenizer(example['question'], example['context'],
                    truncation=True, padding='max_length',
                    max_length=max_len, return_tensors='pt')
    input_ids = enc['input_ids'][0]
    start = min(example['answers']['answer_start'][0], max_len - 1)
    end = min(start + len(example['answers']['text'][0]), max_len - 1)
    return {'input_ids': input_ids, 'start': start, 'end': end}


train_data = [preprocess(e) for e in train_set]
val_data = [preprocess(e) for e in val_set]

def collate(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    start = torch.tensor([b['start'] for b in batch])
    end = torch.tensor([b['end'] for b in batch])
    return input_ids, start, end

train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate)



# Model components
class HyperbolicLinear(nn.Module):
        def __init__(self, in_features, out_features, manifold, c):
            super().__init__()
            self.weight = ManifoldParameter(torch.randn(out_features, in_features) * 0.01, manifold=manifold)
            self.bias = ManifoldParameter(torch.zeros(out_features), manifold=manifold)
            self.manifold = manifold
            self.c = c

        def forward(self, x):
            x = self.manifold.mobius_matvec(self.weight, x)
            x = self.manifold.mobius_add(x, self.bias)
            return x

#Hyperbolic Transformer Blocks
class HyperbolicBERTBlock(nn.Module):
    def __init__(self, c, embed_dim=512, num_heads=2, ff_hidden_dim=4):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.c = c
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        if self.c != 0:
            self.ff1 = HyperbolicLinear(embed_dim, ff_hidden_dim, self.manifold, c)
            self.ff2 = HyperbolicLinear(ff_hidden_dim, embed_dim, self.manifold, c)
        else:
            self.ff1 = nn.Linear(embed_dim, ff_hidden_dim)
            self.ff2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        if self.c != 0:
            attn_out = self.manifold.expmap0(attn_out)
            x = self.manifold.expmap0(x)
            x = self.manifold.mobius_add(x, attn_out)
            x = self.manifold.logmap0(x)
        else:
            x = x + attn_out
        x = self.ln1(x)
        if self.c != 0:
            x_h = self.manifold.expmap0(x)
            x = self.manifold.expmap0(x)
        else:
            x_h = x
        h = F.relu(self.ff1(x_h))
        h = self.ff2(h)
        if self.c != 0:
            h = self.manifold.mobius_add(x, h)
            h = self.manifold.logmap0(h) 
        else:
            x = x + h
        x = self.ln2(x)
        return x


class HyperbolicBERTModel(nn.Module):
    def __init__(self, vocab_size, c, max_len=128, embed_dim=512):
        super().__init__()
        self.c=c
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_len, embed_dim))
        self.encoder_layers = nn.ModuleList([HyperbolicBERTBlock(self.c, embed_dim=embed_dim, num_heads=2, ff_hidden_dim=4) for _ in range(num_layers)])
        #self.bert_block = HyperbolicBERTBlock(embed_dim=embed_dim, num_heads=2, ff_hidden_dim=4, c=c)
        self.qa_output = nn.Linear(embed_dim, 2)  # start & end logits

    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed[:input_ids.size(1)]
        for layer in self.encoder_layers:
            x = layer(x)

        #x = self.bert_block(x)
        logits = self.qa_output(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




def evaluate_on_squad(model, tokenizer, val_set, val_loader, device):
    model.eval()
    predictions = []
    references = []
    for batch_idx, (input_ids, _, _) in enumerate(val_loader):
        input_ids = input_ids.to(device)
        with torch.no_grad():
            start_logits, end_logits = model(input_ids)
        start_pred = torch.argmax(start_logits, dim=1).cpu().tolist()
        end_pred = torch.argmax(end_logits, dim=1).cpu().tolist()

        for i in range(len(input_ids)):
            tokens = input_ids[i].cpu().tolist()
            start = max(0, min(start_pred[i], len(tokens) - 1))
            end = max(start, min(end_pred[i], len(tokens) - 1))
            pred_text = tokenizer.decode(tokens[start:end+1], skip_special_tokens=True).strip()

            example_idx = batch_idx * val_loader.batch_size + i
            if example_idx >= len(val_set):
                continue
            example = val_set[example_idx]
            ref_text = example["answers"]["text"][0]
            ref_start = example["answers"]["answer_start"][0]

            predictions.append({
                "id": example["id"],
                "prediction_text": pred_text
            })
            references.append({
                "id": example["id"],
                "answers": {
                    "text": [ref_text],
                    "answer_start": [ref_start]
                }
            })
    metric = load_metric("squad")
    return metric.compute(predictions=predictions, references=references)

# Training across curvatures
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
curvatures = [0.0, 0.0001, 1.0, 10.0]
num_epochs = 150
results = []
train_error=np.zeros((num_epochs, len(curvatures)))

results=[]

for i,c in enumerate(curvatures):
    print(f"\n======== Training with curvature {c} ========")
    # ball = geoopt.PoincareBall(c=c)
    model = HyperbolicBERTModel(vocab_size=tokenizer.vocab_size, c=c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    def to_one_hot(index, num_classes):
        return F.one_hot(index, num_classes=num_classes).float()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, start, end in train_loader:
            input_ids, start, end = input_ids.to(device), start.to(device), end.to(device)
            optimizer.zero_grad()
    
            start_logits, end_logits = model(input_ids)
    
            # Convert ground truth to one-hot
            seq_len = start_logits.size(1)  # = max_len = 128
            start_onehot = to_one_hot(start, seq_len)
            end_onehot = to_one_hot(end, seq_len)
    
            # Optionally apply softmax to logits for smoother targets
            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)
    
            # MSE Loss
            loss = loss_fn(start_probs, start_onehot) + loss_fn(end_probs, end_onehot)
    
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

        avg_train_loss = total_loss / len(train_loader)
        train_error[epoch, i]=avg_train_loss

        metrics = evaluate_on_squad(model, tokenizer, val_set, val_loader, device)
        em, f1 = metrics["exact_match"], metrics["f1"]
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_train_loss:.4f} | EM: {em:.4f} | F1: {f1:.4f}")
        results.append({
            "curvature": c,
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "EM": em,
            "F1": f1
        })
   

df = pd.DataFrame(results)
#df_pivot = df.pivot(index='epoch', columns='curvature', values=['F1', 'EM'])
df.to_csv(r"/home/iplab/kushal/graph_machine_learning/Hyperbolic_Transformer/squad_scores.csv", index=False)


# df = pd.DataFrame(results.reshape(num_epochs, -1),columns=[f"{metric}_c={c}" for c in curvatures for metric in ["F1", "EM"]])
# df.to_csv(r"/home/mlrl/Sagar_Ghosh_ML_Lab/AAAI_Transformer/Squad/squad_scores.csv", index=False)
print("Saved all evaluation scores to 'squad_eval_scores.csv'")


df_1=pd.DataFrame(train_error, columns=["Test RMSE: 0.0", "Test RMSE: 0.0001", "Test RMSE: 1.0", "Test RMSE: 10.0"])
df_1.to_csv(r"/home/iplab/kushal/graph_machine_learning/Hyperbolic_Transformer/squad_error.csv")