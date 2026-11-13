# Tiny GPT

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/From%20Scratch-Yes-blue?style=for-the-badge" alt="From Scratch"/>
</p>

<p align="center">
  <b>A minimal GPT language model built completely from scratch in PyTorch</b><br>
  Understand every line. No magic. No Hugging Face.
</p>

---

### What is this?

This is a tiny but complete implementation of a **GPT-style transformer** language model.  
It learns to predict the next word from a small custom corpus and can generate text autoregressively.

Perfect for learning how modern LLMs actually work under the hood.

---

### Key Concepts Covered

| Component                  | Description                                      |
|---------------------------|--------------------------------------------------|
| Token Embeddings          | Maps each word to a dense vector                 |
| Positional Embeddings     | Adds order information to tokens                 |
| Multi-Head Self-Attention | Core of the transformer (causal / masked)        |
| Feed-Forward Network      | Position-wise MLP inside each block              |
| Residual + LayerNorm      | Stabilizes training                              |
| Autoregressive Generation | Generates text one token at a time               |

---

### Project Structure

```text
tiny-gpt/
├── transformer_blocks.py   # SelfAttentionHead, MultiHeadAttention, FeedForward, Block
├── train.py                # Model definition, training loop & generation
├── requirements.txt
└── README.md
```

---

### Quick Start

```bash
# Clone the repo
git clone https://github.com/shivamkumar98017/tiny-gpt.git
cd tiny-gpt

# Install dependencies
pip install -r requirements.txt

# Train & generate
python train.py
```

---

### Hyperparameters

| Parameter       | Value     | Description                          |
|-----------------|-----------|--------------------------------------|
| `block_size`    | 6         | Context length (tokens)              |
| `embedding_dim` | 32        | Size of token & position embeddings  |
| `n_heads`       | 2         | Number of attention heads            |
| `n_layers`      | 2         | Number of transformer blocks         |
| `learning_rate` | 1e-3      | AdamW learning rate                  |
| `epochs`        | 1500      | Number of training steps             |
| `batch_size`    | 16        | Batch size                           |

---

### Example Output

```text
my name is shivam <END> the roads of Delhi are busy <END> it is raining in
```

---

### Architecture Overview

```text
Input Tokens
     │
     ▼
┌─────────────────────┐
│  Token Embedding    │
│  + Positional Emb   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Transformer Block  │  × 2 layers
│  ├─ Multi-Head Attn │
│  └─ Feed Forward    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  LayerNorm + Head   │
└─────────────────────┘
     │
     ▼
Next Token Logits
```

---

### Why this project?

Most tutorials jump straight to Hugging Face or large pretrained models.  
This repo forces you to implement every important piece yourself:

- Causal self-attention with masking
- Multi-head attention
- Residual connections + LayerNorm
- Simple training loop
- Greedy / multinomial sampling for generation

After reading this code, the internals of GPT-2 / GPT-3 / Llama will feel much less mysterious.

---

### Future Ideas

- [ ] Add temperature & top-k sampling
- [ ] Save / load checkpoints
- [ ] Character-level version
- [ ] Increase model size and train on a bigger corpus
- [ ] Add a simple web UI for generation

---

### Author

**Shivam Kumar**  
AI/ML Engineer • Cybersecurity Enthusiast  

[GitHub](https://github.com/shivamkumar98017) • [Instagram](https://www.instagram.com/shivamkumar98017/)

---

<p align="center">
  Made with ❤️ and PyTorch
</p>
