import torch
from torch import nn
from datasets import load_dataset


def build_dataset(sample_size=2000):
    """Load part of Wikitext-2 dataset and build vocabulary."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{sample_size}]")
    text = "\n".join(ds["text"])
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos


def get_batch(data, block_size, batch_size=1):
    """Generate a random batch."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).t()
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).t()
    return x, y


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size=128, nhead=4, nhid=256, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=nhid)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, src):
        # src shape: [seq_len, batch]
        src = self.embed(src) * (self.embed.embedding_dim ** 0.5)
        out = self.transformer(src)
        out = self.fc(out)
        return out


def train(num_steps=200):
    block_size = 128
    data, stoi, itos = build_dataset()
    model = SimpleTransformer(len(stoi))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(num_steps):
        x, y = get_batch(data, block_size)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    train()
