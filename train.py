import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformer_blocks import Block


# ----------------------------
# Config
# ----------------------------
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500
batch_size = 16
checkpoint_path = "tiny_gpt_checkpoint.pt"


# ----------------------------
# Corpus
# ----------------------------
corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "my name is shivam",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match",
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)

words = list(set(text.split()))
vocab_size = len(words)

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

print(f"Vocab size : {vocab_size}")
print(f"Data length: {len(data)}")
print(f"Torch      : {torch.__version__}")
print(f"CUDA       : {torch.cuda.is_available()}")


def get_batch(batch_size=batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ----------------------------
# Model
# ----------------------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.

        Args:
            idx: (B, T) starting context indices
            max_new_tokens: number of new tokens to generate
            temperature: >1.0 = more random, <1.0 = more deterministic (default 1.0)
            top_k: if set, only sample from the top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # scale by temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def save_checkpoint(model, optimizer, step, loss, path=checkpoint_path):
    """Save model + optimizer state so training can be resumed later."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "vocab_size": vocab_size,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "config": {
            "block_size": block_size,
            "embedding_dim": embedding_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
        },
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer=None, path=checkpoint_path):
    """Load a previously saved checkpoint."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return None

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {path} (step={checkpoint.get('step')}, loss={checkpoint.get('loss'):.4f})")
    return checkpoint


# ----------------------------
# Training
# ----------------------------
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Uncomment the next line if you want to resume from a previous checkpoint
# load_checkpoint(model, optimizer)

print("\nTraining started...")
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0 or step == epochs - 1:
        print(f"Step {step:4d} | loss = {loss.item():.4f}")

print("Training finished!\n")

# Save final checkpoint
save_checkpoint(model, optimizer, step=epochs - 1, loss=loss.item())


# ----------------------------
# Generation
# ----------------------------
context = torch.tensor(
    [[word2idx["my"], word2idx["name"], word2idx["is"], word2idx["shivam"]]],
    dtype=torch.long,
)

# Default generation (temperature=1.0, no top-k)
out = model.generate(context, max_new_tokens=15)
print("Generated text (default):")
print(" ".join(idx2word[int(i)] for i in out[0]))

# More creative / diverse generation
out_creative = model.generate(context.clone(), max_new_tokens=15, temperature=1.2, top_k=5)
print("\nGenerated text (temp=1.2, top_k=5):")
print(" ".join(idx2word[int(i)] for i in out_creative[0]))

# More focused / deterministic generation
out_focused = model.generate(context.clone(), max_new_tokens=15, temperature=0.7, top_k=3)
print("\nGenerated text (temp=0.7, top_k=3):")
print(" ".join(idx2word[int(i)] for i in out_focused[0]))
