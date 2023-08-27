import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    DistributedSampler,
)


# Hyperparameters
batch_size = 100  # B
block_size = 200  # T
emb_size = 128  # C
num_blocks = 4
num_heads = 4
head_size = 256
dropout = 0.2

data_dir = os.path.join(
    os.path.dirname(os.getcwd()), "Data/Tiny shakespeare/input.txt"
)

with open(data_dir, "r") as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(sorted(list(set(text))))
data_size = len(text)

if torch.cuda.is_available():
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"

token_encodings = {}
token_decodings = {}
for i, token in enumerate(vocab):
    token_encodings[token] = i
    token_decodings[i] = token


def encode(txt):
    enc_char = [token_encodings[char] for char in txt]
    return enc_char


def decode(enc_tokens):
    dec_char = [token_decodings[idx] for idx in enc_tokens]
    decoded_str = "".join(dec_char)
    return decoded_str


def generate_batch(batch_size, block_size):
    idx = torch.randint(0, data_size - block_size - 1, (batch_size,))
    data = torch.tensor(
        [encode(text[i : i + block_size]) for i in idx], device=device
    )  # B x T
    targets = torch.tensor(
        [encode(text[i + 1 : i + block_size + 1]) for i in idx], device=device
    )  # B x T
    return data, targets


class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data = open(data_dir, "r").read()
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.data,
            [
                int(len(self.data) * 0.8),
                len(self.data) - int(len(self.data) * 0.8),
            ],
        )
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = val_dataset

    def __getitem__(self, idx):
        data = torch.tensor(
            [encode(self.dataset[i : i + block_size]) for i in idx],
            device=device,
        )  # B x T
        targets = torch.tensor(
            [encode(self.dataset[i + 1 : i + block_size + 1]) for i in idx],
            device=device,
        )  # B x T
        return data, targets

    def __len__(self):
        return len(self.dataset)


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.q = nn.Linear(emb_size, self.head_size, device=device)
        self.k = nn.Linear(emb_size, self.head_size, device=device)
        self.v = nn.Linear(emb_size, self.head_size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q(x)  # B, T, C -> B, T, H
        k = self.k(x)
        v = self.v(x)
        B, T, H = q.shape
        wei = (
            q @ k.transpose(-1, -2) / np.sqrt(self.head_size)
        )  # B, T, H @ B, H, T -> B, T, T
        mask = torch.tril(torch.ones(B, T, T)).to(device)
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # B, T, H
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, emb_size):
        super().__init__()
        self.n_heads = n_heads
        self.emb_size = emb_size
        self.head_size = emb_size // n_heads
        self.linear = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(0.2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = []
        for i in range(self.n_heads):
            att_head = SelfAttention(self.head_size)
            out.append(att_head(x))
        # print(len(out), out[0].shape)
        logits = torch.cat(out, dim=-1)
        logits = self.linear(logits)
        return logits


class FeedForwardBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.mha = MultiHeadedAttention(num_heads, emb_size)
        self.ff_net = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout),
        )
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.layer_norm_2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.mha(self.layer_norm_1(x))  # B, T, C
        x = x + self.ff_net(self.layer_norm_2(x))  # B, T, vocab_size
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, emb_size, device=device)
        self.pos_emb_table = nn.Embedding(block_size, emb_size, device=device)
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.final_ll = nn.Linear(emb_size, vocab_size, device=device)
        self.blocks = nn.Sequential(
            *[FeedForwardBlock(num_heads) for _ in range(num_blocks)]
        )

    def forward(self, x, targets=None):
        token_emb = self.token_emb_table(x)  # B, T, C
        pos_emb = self.pos_emb_table(
            torch.arange(x.shape[-1], device=device)
        )  # T, C
        x = token_emb + pos_emb  # B, T, C
        x = self.blocks(x)
        logits = self.final_ll(x)
        B, T, C = logits.shape
        if targets is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_slice = idx[:, -block_size:]
            logits, loss = self.forward(idx_slice)
            logits = logits[:, -1, :]
            probabs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return decode(idx[0].tolist())

    def train(self, num_steps, batch_size):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, betas=(0.9, 0.95)
        )
        loss_ar = []
        for step in range(num_steps):
            optimizer.zero_grad()
            data, targets = generate_batch(batch_size, block_size)
            logits, loss = self.forward(data, targets)
            loss_ar.append(loss.item())
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print(f"Step {step}, loss {loss.item()}")
        return loss_ar
