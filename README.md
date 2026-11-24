<div align="center">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="90">

# Tiny GPT

**A minimal GPT language model built completely from scratch in PyTorch**  
Understand every line. No magic. No Hugging Face.

<br>

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![From Scratch](https://img.shields.io/badge/From%20Scratch-Yes-3B82F6?style=for-the-badge)
![Educational](https://img.shields.io/badge/Educational-Yes-F59E0B?style=for-the-badge)

</div>

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
| Temperature & Top-k       | Controllable sampling for creative vs focused output |
| Checkpoint Save / Load    | Resume training or generate from a saved model   |

---

### Project Structure

```text
tiny-gpt/
├── transformer_blocks.py   # SelfAttentionHead, MultiHeadAttention, FeedForward, Block
├── train.py                # Model definition, training loop, generation & checkpoints
├── requirements.txt
├── .gitignore
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

# Train & generate (also saves a checkpoint)
python train.py
```

After training finishes, a file `tiny_gpt_checkpoint.pt` is created.  
You can later load it and continue training or just generate text.

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

### Generation Controls

The `generate()` method supports **temperature** and **top-k** sampling:

```python
# Default (temperature=1.0, no top-k)
out = model.generate(context, max_new_tokens=15)

# More creative / diverse
out = model.generate(context, max_new_tokens=15, temperature=1.2, top_k=5)

# More focused / deterministic
out = model.generate(context, max_new_tokens=15, temperature=0.7, top_k=3)
```

- **temperature > 1.0** → more random / creative
- **temperature < 1.0** → more deterministic / focused
- **top_k** → only sample from the top-k most likely tokens

---

### Checkpointing

After training, the script automatically saves:

```text
tiny_gpt_checkpoint.pt
```

It contains:
- Model weights
- Optimizer state
- Vocabulary mappings
- Training step & loss

To resume training later, just uncomment this line near the top of the training section in `train.py`:

```python
load_checkpoint(model, optimizer)
```

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
- Temperature + top-k sampling for generation
- Checkpoint save & load

After reading this code, the internals of GPT-2 / GPT-3 / Llama will feel much less mysterious.

---

### Future Ideas

- [x] Add temperature & top-k sampling
- [x] Save / load checkpoints
- [ ] Character-level version
- [ ] Increase model size and train on a bigger corpus
- [ ] Add a simple web UI for generation

---

<div align="center">

**Shivam Kumar**  
AI/ML Engineer • Cybersecurity Enthusiast  
Motihari, Bihar

[GitHub](https://github.com/shivamkumar98017) · [Instagram](https://www.instagram.com/shivamkumar98017/) · [Email](mailto:shivamk34442@gmail.com)

<br>

<sub>Made with ❤️ and pure PyTorch</sub>

</div>

---

## 🌐 Connect With Me

- **Instagram:** [www.instagram.com/codes.flow](https://www.instagram.com/codes.flow/)
- **LinkedIn:** [www.linkedin.com/in/shivam-kumar-b24677431](https://www.linkedin.com/in/shivam-kumar-b24677431/)
- **X:** [x.com/shivamk344007](https://x.com/shivamk344007)
