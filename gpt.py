import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters for the model 
batch_size = 32 # how many individual sequences to process in parallel 
block_size = 128 # maximum context length for prediction 
max_iters = 5000 # maximum number of iterations 
eval_interval = 500 # how often to evaluate the model on the validation set 
eval_iters = 200
n_embd = 64 # embedding dimensionality 
n_head = 6 # number of attention heads
n_layer = 6 # number of transformer layers 
dropout = 0.2 # dropout rate 
learning_rate = 3e-4 # learning rate 
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
# -------- 

torch.manual_seed(1337) 

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in this text 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers and vice versa 
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers 
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string 

# train and test splits 
data = torch.tensor(encode(text), dtype=torch.long) 
n = int(0.9 * len(data)) # first 90% will be train, rest val 
train_data = data[:n] 
val_data = data[n:]

# data loading 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y 
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) 
    return x, y 

@torch.no_grad() # this context manager disables gradient tracking
def estimate_loss(): 
    """ 
    averages out the loss over multiple iterations 
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module): 
    """
    one head of self-attention, which means it computes the attention scores between all pairs of tokens in a sequence. 
    attention scores are computed using the dot product of the query and key vectors. The resulting scores are then normalized by dividing by the square root of the head size. 
    The softmax function is applied to these scores to obtain a probability distribution over all pairs of tokens. This probability distribution is then used to compute the weighted sum of the value vectors, which gives the output of the attention head.
    """

    def __init__(self, head_size): 
        super().__init__() # initialize the layers and buffers
        self.key = nn.Linear(n_embd, head_size, bias=False) # key linear layer for computing attention scores
        self.query = nn.Linear(n_embd, head_size, bias=False) # query linear layer 
        self.value = nn.Linear(n_embd, head_size, bias=False) # value linear layer 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix for masking future tokens

        self.dropout = nn.Dropout(dropout) # dropout layer for regularization 

    def forward(self, x): 
        # input of size (batch, time-step, channels) 
        # output of size (batch, time-step, channels) 
        B,T,C = x.shape # batch size, sequence length, embedding dimension
        k = self.key(x) # compute key vectors 
        q = self.query(x) # compute query vectors 
        # compute attention scores (B, T, hs) * (B, hs, T) -> (B, T, T) ("affinities")
        wei = torch.bmm(q, k.transpose(-2,-1)) * k.shape[-1]**-0.5 # scale by sqrt(d_k) for numerical stability. (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # mask future tokens. (B, T, T)
        wei = F.softmax(wei, dim=-1) # apply softmax to get attention weights. (B, T, T)
        wei = self.dropout(wei) # apply dropout for regularization 
        # perform weighted sum of values using attention weights 
        v = self.value(x) # compute value vectors 
        out = torch.bmm(wei, v) # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module): 
    """
    multi-head self-attention layer that consists of multiple heads. Each head computes its own set of attention scores between all pairs of tokens in a sequence. The outputs from all heads are then concatenated to form the final output.
    """

    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # create a list of heads
        self.proj = nn.Linear(head_size * num_heads, n_embd) # project the concatenated outputs from all heads to the original size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        out = torch.cat([head(x) for head in self.heads], dim=-1) # concatenate the outputs from all heads along the last dimension
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token embedding table 
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # position embedding table
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

 

    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # add positional embeddings to token embeddings. (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
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
        #idx is a (B,T) tensor of indices in the current context 
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens 
            idx_cond = idx[:, -block_size:] # (B,C)
            # get predictions 
            logits, loss = self(idx_cond)
            # focus only on the last time-step 
            logits = logits[:, -1, :] # (B,C) 
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the end of context 
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss() # for giving us the avg value
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) 
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))