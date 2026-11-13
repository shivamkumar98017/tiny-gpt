# Tiny GPT

A minimal GPT (Generative Pre-trained Transformer) language model implemented **from scratch** in PyTorch.

This project demonstrates the core concepts of modern LLMs:
- Token + Positional Embeddings
- Multi-Head Self-Attention
- Transformer Blocks
- Autoregressive text generation

## Features

- Extremely small & easy to understand codebase
- Trains on a custom Hindi-English mixed corpus
- Generates text starting from a given prompt
- Fully from-scratch implementation (no Hugging Face / transformers library)

## Project Structure

```
tiny-gpt/
├── transformer_blocks.py   # Self-Attention, Multi-Head Attention, FeedForward, Block
├── train.py                # Model definition + training + generation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

## Hyperparameters

| Parameter       | Value |
|-----------------|-------|
| block_size      | 6     |
| embedding_dim   | 32    |
| n_heads         | 2     |
| n_layers        | 2     |
| learning_rate   | 1e-3  |
| epochs          | 1500  |

## Example Output

```
my name is shivam <END> the roads of Delhi are busy <END> it is raining in
```

## Author

**Shivam Kumar**  
[GitHub](https://github.com/shivamkumar98017)
