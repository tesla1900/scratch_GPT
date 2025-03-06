import torch
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters for the model 
batch_size = 64 # how many individual sequences to process in parallel 
block_size = 256 # maximum context length for prediction 
max_iter = 5000 # maximum number of iterations 
eval_interval = 500 # how often to evaluate the model on the validation set 
eval_iters = 200
n_embd = 384 # embedding dimensionality 
n_head = 6 # number of attention heads
n_layer = 6 # number of transformer layers 
dropout = 0.2 # dropout rate 
learning_rate = 1e-3 # learning rate 

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
    """one head of self-attention, which means it computes the attention scores between all pairs of tokens in a sequence. 
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
        


