# Copyright (c) 2023 Juan M. G. de Agüero
# https://colab.research.google.com/drive/1JdRdVvtYIRehVCTJoTFPUWw93ajK8Asg?usp=sharing

import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", 'r') as file:
  text = file.read()

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = dict()
itos = dict()
for i in range(len(chars)):
  stoi[chars[i]] = i
for i in range(len(chars)):
  itos[i] = chars[i]

def encode(string):
  encoding = []
  for char in string:
    encoding.append(stoi[char])
  return encoding

def decode(integers):
  string = ""
  for integer in integers:
    string += itos[integer]
  return string

string = "hello there"
print(encode(string))
print(decode(encode(string)))

data = encode(text)
print(len(data))
print(data[:30])

n = int(0.9 * len(data))
train_data = torch.tensor(data[:n], dtype=torch.long)
val_data = torch.tensor(data[n:], dtype=torch.long)
print(train_data[:200])

block_size = 8
print(train_data[: block_size + 1])

torch.manual_seed(1337)
batch_size = 32
block_size = 8

def get_batch(split):

  if split == "train":
    data = train_data
  else:
    data = val_data

  ix_data = []
  for _ in range(batch_size):
    ix_data.append(torch.randint(data.numel() - block_size, (1,)).item())
  ix = torch.tensor(ix_data, dtype=torch.int32)

  lst = []
  n = 0
  for i in range(ix.numel()):
    n = ix[i] + block_size - ix[i]
    for j in range(ix[i], ix[i] + block_size):
      lst.append(data[j].item())
  x = torch.tensor(lst)
  x = x.reshape(ix.numel(), n)

  lst = []
  for i in range(ix.numel()):
    n = (ix[i] + block_size + 1) - (ix[i] + 1)
    for j in range(ix[i] + 1, ix[i] + block_size + 1):
      lst.append(data[j].item())
  y = torch.tensor(lst)
  y = y.reshape(ix.numel(), n)

  return (x, y)

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb[:8])
print("targets:")
print(yb.shape)
print(yb[:8])

for b in range(batch_size):
  for t in range(block_size):
    context = xb[b, : t + 1]
    target = yb[b, t]
    #print(str(context.tolist()) + ": " + str(target.item()))

def mean(arr):
  sum = torch.sum(arr, dim=0, keepdim=True)
  return torch.div(sum, torch.tensor([arr.shape[0]]))

def softmax(logits):
  max_logits = torch.index_select(torch.max(logits, dim=1, keepdim=True).values, dim=1, index=torch.tensor([0]))
  exp_logits = torch.exp(torch.sub(logits, max_logits))
  sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
  return torch.div(exp_logits, sum_exp_logits)

def cross_entropy(logits, targets):
  return torch.mean(torch.neg(torch.log(torch.gather(softmax(logits), 1, torch.unsqueeze(targets, 1)) + 1e-10)))

def multinomial(probs, num_samples):
  pass

def create_pattern(block_size, T):
  return [i for i in range(block_size)] * (T // block_size) + [i for i in range(T % block_size)]

num_embedding = 32
token_embedding_table = torch.randn(vocab_size, num_embedding, requires_grad=True)
position_embedding_table = torch.randn(block_size, num_embedding, requires_grad=True)
lm_head_weight = torch.randn(num_embedding, vocab_size, requires_grad=True)
lm_head_bias = torch.randn(vocab_size, requires_grad=True)

def forward(index, targets=None):
  B, T = index.shape
  total_elements = index.shape[0] * index.shape[1]
  token_embedding = torch.reshape(torch.index_select(token_embedding_table, 0, torch.reshape(index, (total_elements,))), (B, T, num_embedding))
  pattern = create_pattern(block_size, T)
  position_embedding_indices = torch.tensor(pattern)
  position_embedding = torch.index_select(position_embedding_table, 0, position_embedding_indices)
  x = torch.add(token_embedding, position_embedding)
  logits = torch.add(torch.matmul(x, lm_head_weight), lm_head_bias)
  if targets is None:
    loss = None
  else:
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = cross_entropy(logits, targets)
  return logits, loss

def generate(index, max_new_tokens):
  for _ in range(max_new_tokens):
    logits, loss = forward(index)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    index_next = torch.multinomial(probs, num_samples=1)
    index = torch.cat((index, index_next), dim=1)
  return index

logits, loss = forward(xb, yb)
print(logits.shape)
print(loss)

index = torch.zeros((1, 1), dtype=torch.long)
print(decode(generate(index, max_new_tokens=100)[0].tolist()))

learning_rate = 1e-3
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
weight_decay = 0.01
m = 0
v = 0
t = 0

for step in range(10000):
  t += 1
  xb, yb = get_batch("train")
  _, loss = forward(xb, yb)
  if token_embedding_table.grad is not None: token_embedding_table.grad.zero_()
  loss.backward()
  g = token_embedding_table.grad.detach()
  m = beta1 * m + (1 - beta1) * g
  v = beta2 * v + (1 - beta2) * g * g
  m_hat = m / (1 - beta1 ** t)
  v_hat = v / (1 - beta2 ** t)
  token_embedding_table = token_embedding_table.detach() - learning_rate * (m_hat / (v_hat.sqrt() + epsilon) + weight_decay * token_embedding_table.detach())
  token_embedding_table.requires_grad = True

print(loss.item())
index = torch.zeros((1, 1), dtype=torch.long)
print(decode(generate(index, max_new_tokens=500)[0].tolist()))