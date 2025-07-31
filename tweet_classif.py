import geoopt
import torch
import numpy as np
import pandas as pd
import os

import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from geoopt import PoincareBall

from transformers import BertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torch.optim import Adam
from transformers import AutoTokenizer

from language_transformer import Transformer



def load_tweetqa_data(model_name="bert-base-uncased", max_len=128, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("ucsbnlp/tweet_qa")

    def preprocess(examples):
        inputs = tokenizer(examples['Question'], examples['Tweet'], truncation=True, padding="max_length", max_length=max_len)
        inputs['label'] = [0 if (isinstance(ans, list) and len(ans) > 0 and ans[0].strip().lower() in ['no', 'false']) else 1 for ans in examples['Answer']]


        #inputs['label'] = [0 if ans[0].lower() in ['no', 'false'] else 1 for ans in examples['Answer']]
        return inputs

    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size)

    return train_loader, val_loader, tokenizer


train_loader, val_loader, tokenizer = load_tweetqa_data()

# for batch in train_loader:
#     print(batch['label'])
# exit()

curvatures = [0.0001, 1.0, 10.0]
DATASET = 'tweet_qa'

for c in curvatures:
    print(f'Curvature: {c:.4f}')
    # Create model
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=2,
        d_model=128,
        n_heads=4,
        n_layers=6,
        d_ff=512,
        max_seq_length=128,
        c = c,
        dropout=0.1
    )
    # print(model)

    def train(model, train_loader, val_loader, tokenizer, epochs=3, lr=1e-3, device="cuda:0"):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='mean')
        
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                labels = F.one_hot(labels, num_classes=2).float()
                # print(logits.shape, "\t", labels.shape)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            test_accuracy = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, attention_mask)
                    labels = F.one_hot(labels, num_classes=2).float()
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    pred_classes = logits.argmax(dim=-1)
                    labels = batch["label"].to(device)
                    acc = (pred_classes == labels).sum().item()
                    acc /= (pred_classes.shape[0])
                    test_accuracy += acc
                

            avg_val_loss = total_val_loss / len(val_loader)
            test_accuracy /= len(val_loader)
            val_losses.append(avg_val_loss)

            print(f'Epoch: {epoch+1} || Train loss {avg_train_loss:.4f} || Val loss {avg_val_loss:.4f} || Test Acc {test_accuracy*100:.4f}')

        return train_losses, val_losses

    train_losses, val_losses = train(model, train_loader, val_loader, tokenizer, epochs=100)

    filename = '/curvatures/'+ DATASET +'_curvature_results_'+ str(c) +'.csv'
    df = pd.DataFrame({
        'test': val_losses
    })

    # Save to CSV
    df.to_csv(os.getcwd() + filename, index=False)

