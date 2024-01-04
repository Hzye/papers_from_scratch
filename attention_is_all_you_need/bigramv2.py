import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm

# hyperparams
batch_size = 32
block_size = 8
n_iters = 5000
eval_interval = 300
eval_n_iters = 200
learning_rate = 1e-3
n_embeds = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# load data
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

unique_chars_list = sorted(list(set(text)))
unique_chars_str = "".join(unique_chars_list)

char_counts = [text.count(char) for char in unique_chars_list]
df_chars = pd.DataFrame({"char": unique_chars_list, "counts": char_counts}).sort_values(by=["counts"], ascending=False)

enc_table = {char: i  for i, char in enumerate(unique_chars_list)}
dec_table = {i: char for i, char in enumerate(unique_chars_list)}

enc = lambda string: [enc_table[char] for char in string]
dec = lambda lst: "".join([dec_table[i] for i in lst])

# tokenise the data
data = torch.tensor(enc(text), dtype=torch.long)

# split data
train_test_ratio = 0.9
train_len = int(train_test_ratio*len(data))
test_len = len(data) - train_len

train_data = data[:train_len]
test_data = data[train_len:]


# dataloader
def get_random_batch(dset):
    data = train_data if dset == "train" else test_data
    block_start_idx = torch.randint(low=0, high=len(data)-1-block_size, size=[batch_size,])
    samples = torch.vstack([data[start_idx:start_idx+block_size] for start_idx in block_start_idx])
    labels = torch.vstack([data[start_idx+1:start_idx+block_size+1] for start_idx in block_start_idx])
    samples, labels = samples.to(device), labels.to(device)
    return samples, labels

# from Andrej's video
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_n_iters)
        for i in range(eval_n_iters):
            X, Y = get_random_batch(split)
            _, loss = m(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# from Andrej's video
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(block_size, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape

        # idx and targets are both (batch,time) == (4,8) tensor of integers
        tok_embs = self.token_embedding_table(idx) # (batch,time,channel==n_embeds)
        pos_embs = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_embs + pos_embs # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (batch,time) array of indices in the current block
        for _ in range(max_new_tokens):
            # get preds
            logits, loss = self(idx, None)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (batch,channel)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (batch,channel)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch,1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch,time+1)
        return idx

vocab_size = len(unique_chars_str)
m = BigramLanguageModel().to(device)

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# train
train_losses = []
for iter in range(n_iters):

    # eval
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batches of data
    xb, yb = get_random_batch("train")

    # eval
    logits, loss = m(xb, yb)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

# generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(dec(m.generate(context, max_new_tokens=300)[0].tolist()))