# Copyright (c) 2023 Juan M. G. de Agüero
# https://colab.research.google.com/drive/1JdRdVvtYIRehVCTJoTFPUWw93ajK8Asg?usp=sharing

import torch

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
batch_size = 4
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

batch = get_batch("train")
xb = batch[0]
yb = batch[1]
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

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

torch.manual_seed(1337)

class BiagramLM(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, index, targets):
    logits = self.token_embedding_table(index)
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = cross_entropy(logits, targets)
    return logits, loss

m = BiagramLM(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)