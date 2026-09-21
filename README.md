# Universal Consistency of Transformers as Functional Regressors

This repository contains the implementation and experiments used to study **hyperbolic Transformer architectures** and their behavior relative to their standard Euclidean counterparts across a range of sequence, vision, and graph tasks.

Every model in this repository is built around a single design choice: any linear sub-layer (feed-forward blocks, projection heads) can either operate in ordinary Euclidean space or be replaced by a **Möbius/hyperbolic linear layer** defined on a Poincaré ball of curvature $c$. Setting $c = 0$ recovers the standard Euclidean Transformer, while $c > 0$ places the same computation on a hyperbolic manifold. All experiments are run across a shared sweep of curvatures so that the Euclidean model and several hyperbolic variants can be compared directly under identical training conditions.

The experiments cover four task families:

1. **Sequence-to-sequence functional regression / language modeling** — a full encoder-decoder Transformer (`language_transformer.py`).
2. **Image classification** — a Vision Transformer (ViT) trained on MNIST, Fashion-MNIST, and CIFAR-10 (`vision_transformer.py`, `vision_main.py`).
3. **Extractive question answering** — a hyperbolic BERT-style span predictor trained and evaluated on SQuAD (`squad_hyp_4.py`) and TweetQA (`tweet_classif.py`).
4. **Graph node classification** — a Graph Transformer with hyperbolic layers evaluated on the SBM `PATTERN` / `CLUSTER` benchmarks (`GT/`).

---

## Repository Structure


> Datasets, checkpoints, and generated result CSVs are not tracked in this repository and are produced locally when the scripts are run.

---

## 1. Motivation

[#1-motivation](#1-motivation)

Standard Transformers operate entirely in Euclidean space. Hyperbolic geometry has been argued to represent hierarchical and tree-like structure more efficiently, but it is unclear whether swapping Euclidean sub-layers for hyperbolic ones changes the **functional behavior** of a Transformer in practice. This repository implements matched Euclidean/hyperbolic pairs of the same architecture — differing only in the curvature $c$ of the manifold used inside the feed-forward sub-layers — and evaluates them side by side on sequence, vision, and graph tasks.

The curvature parameter is swept over the same grid in every experiment:

$$c \in \{0.0,\ 0.0001,\ 1.0,\ 10.0\}$$

with $c = 0$ corresponding to the ordinary Euclidean model.

---

## 2. Hyperbolic Building Blocks

[#2-hyperbolic-building-blocks](#2-hyperbolic-building-blocks)

All hyperbolic components are built on top of [`geoopt`](https://github.com/geoopt/geoopt)'s `PoincareBall` manifold. The core primitive shared across every model is:

```python
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, c):
        ...
    def forward(self, x):
        x = self.manifold.mobius_matvec(self.weight, x)
        x = self.manifold.mobius_add(x, self.bias)
        return x
```

Inside each Transformer block, residual connections and feed-forward sub-layers switch between Euclidean addition and Möbius addition (`mobius_add`) depending on whether $c = 0$ or $c > 0$, with `expmap0` / `logmap0` used to move activations onto and off of the manifold's tangent space at the origin.

---

## 3. Sequence-to-Sequence Transformer

[#3-sequence-to-sequence-transformer](#3-sequence-to-sequence-transformer)

`language_transformer.py` implements a complete encoder-decoder Transformer:

- Multi-head self-attention (`MultiHeadAttention`) and sinusoidal `PositionalEncoding`, both standard Euclidean.
- A `PositionwiseFeedForward` block that uses `HyperbolicLinear` layers when $c \neq 0$ and ordinary `nn.Linear` layers when $c = 0$.
- `EncoderLayer` / `DecoderLayer` residual connections implemented via `mobius_add` on the Poincaré ball when hyperbolic, or standard addition otherwise.
- A configurable `Transformer` wrapper exposing `d_model`, `n_heads`, `n_layers`, `d_ff`, and curvature `c`.

This module is imported directly by `tweet_classif.py` and mirrors the block design reused (in a lighter form) by `squad_hyp_4.py`.

---

## 4. Vision Transformer

[#4-vision-transformer](#4-vision-transformer)

`vision_transformer.py` adapts the same hyperbolic/Euclidean design to image classification via patch embeddings, and `vision_main.py` drives training and evaluation.

### Datasets

[#datasets](#datasets)

- MNIST
- Fashion-MNIST
- CIFAR-10

All images are resized to $32\times32$ and split into $4\times4$ patches.

### Training configuration

[#training-configuration](#training-configuration)

- Batch size: `64`
- Epochs: `150`
- Learning rate: `1e-4` (AdamW, weight decay `1e-4`, cosine annealing schedule)
- Embedding dimension: `256`, depth `6`, `8` attention heads
- Curvatures swept: `[0.0, 0.0001, 1.0, 10.0]`

### Run

[#run](#run)


The best checkpoint per curvature is saved as `best_vit_<dataset>.pth`, and per-epoch test loss is written to `<dataset>_curvature_results_<c>.csv`.

---

## 5. Extractive Question Answering (SQuAD)

[#5-extractive-question-answering-squad](#5-extractive-question-answering-squad)

`squad_hyp_4.py` trains a lightweight hyperbolic BERT-style encoder (`HyperbolicBERTModel`, built from `HyperbolicBERTBlock` layers) directly on the SQuAD dataset for start/end span prediction.

- Tokenizer: `bert-base-uncased` (`BertTokenizerFast`)
- Sequence length: `128`
- Encoder depth: `3` layers
- Metric: Exact Match (EM) and F1, computed via `evaluate.load("squad")`
- Curvatures swept: `[0.0, 0.0001, 1.0, 10.0]`, `150` epochs each

### Run

[#run-1](#run-1)


> The script writes evaluation and training-loss curves to hardcoded local paths (`squad_scores.csv`, `squad_error.csv`). Update these paths before running on another machine.

---

## 6. Tweet Question Answering (TweetQA)

[#6-tweet-question-answering-tweetqa](#6-tweet-question-answering-tweetqa)

`tweet_classif.py` reuses the encoder-decoder `Transformer` from `language_transformer.py`, configured as a binary classifier, and trains it on the [TweetQA](https://huggingface.co/datasets/ucsbnlp/tweet_qa) dataset (question + tweet → yes/no-style answer label).

- Tokenizer: `bert-base-uncased`
- Curvatures swept: `[0.0001, 1.0, 10.0]`

### Run

[#run-2](#run-2)


---

## 7. Graph Transformer for Node Classification

[#7-graph-transformer-for-node-classification](#7-graph-transformer-for-node-classification)

The `GT/` directory contains a Graph Transformer with hyperbolic layers (`hyp_layers.py`), built on [DGL](https://www.dgl.ai/), evaluated on the SBM `PATTERN` and `CLUSTER` node-classification benchmarks.

Key files:

- `graph_transformer_layer.py` / `graph_transformer_edge_layer.py` — Transformer layers with and without edge features.
- `hyp_layers.py` — hyperbolic (`HNNLayer`) primitives and curvature utilities shared with the Poincaré-ball / Hyperboloid manifolds.
- `graph_transformer_net.py`, `load_net.py`, `mlp_readout_layer.py` — model assembly and readout heads.
- `main_SBMs_node_classification.py` — training entry point (config-driven via JSON, with `--c` controlling curvature and `--seed` the random seed).
- `run_sbms.sh` — example invocations sweeping curvature, positional encoding (`LapPE` / `RWPE`), and normalization (`BN` / `LN`) across `PATTERN` and `CLUSTER`.

### Run

[#run-3](#run-3)



> This module expects `nets/SBMs_node_classification/load_net.py` and `data/data.py` on the Python path (the standard [`graphdeeplearning/graphtransformer`](https://github.com/graphdeeplearning/graphtransformer) benchmark layout) along with the corresponding dataset config `.json` files.

---

## 8. Software Requirements

[#8-software-requirements](#8-software-requirements)

The experiments are implemented in Python using:

- Python 3
- PyTorch, TorchVision
- [`geoopt`](https://github.com/geoopt/geoopt) (Riemannian/hyperbolic optimization and manifolds)
- Hugging Face `transformers`, `datasets`, `evaluate`
- [DGL](https://www.dgl.ai/) (for the `GT/` graph experiments)
- NumPy, Pandas, Matplotlib, tqdm

A typical environment can be installed with:



---

## 9. Reproducibility

[#9-reproducibility](#9-reproducibility)

Each script sweeps the same curvature grid so that the Euclidean baseline ($c=0$) and hyperbolic variants ($c>0$) are trained under matched hyperparameters, optimizer settings, and epoch budgets. Random seeds are set where applicable, but exact reproducibility may still depend on the hardware backend (CPU, CUDA, or Apple MPS) and the corresponding PyTorch build.

> **Note:** Several scripts (`squad_hyp_4.py`, `vision_main.py`) contain local absolute output paths for result CSVs. Update these paths before running on another machine.



