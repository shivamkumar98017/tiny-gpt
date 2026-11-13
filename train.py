import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# ----------------------------
# Training
# ----------------------------
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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


# ----------------------------
# Generation
# ----------------------------
context = torch.tensor(
    [[word2idx["my"], word2idx["name"], word2idx["is"], word2idx["shivam"]]],
    dtype=torch.long,
)
out = model.generate(context, max_new_tokens=15)

print("Generated text:")
print(" ".join(idx2word[int(i)] for i in out[0]))
