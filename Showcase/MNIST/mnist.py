# Copyright (c) 2023 Juan M. G. de Agüero
# https://colab.research.google.com/drive/1GwRjaX5Jh4rTxrPH9ChfaPl-YTaznoIn?usp=sharing

import numpy as np
import tensorflow.keras as keras
import torch

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

n = 6000
x_train = x_train[:n, :]
y_train = y_train[:n]
x_test = x_test[:n, :]
y_test = y_test[:n]

x_train = x_train / 255
x_test = x_test / 255

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

x = 11
for i in range(28):
  for j in range(28):
    if x_test[x][i * 28 + j] > 0:
      print("00", end='')
    else:
      print("..", end='')
  print()
print(y_test[x])

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

w1 = torch.randn(784, 128, requires_grad=True, dtype=torch.float32)
b1 = torch.randn(128, requires_grad=True, dtype=torch.float32)
w2 = torch.randn(128, 10, requires_grad=True, dtype=torch.float32)
b2 = torch.randn(10, requires_grad=True, dtype=torch.float32)

learning_rate = 0.1

for epoch in range(100):
  a = torch.relu(torch.add(torch.matmul(x_train, w1), b1))
  yPred = torch.add(torch.matmul(a, w2) , b2)
  loss = cross_entropy(yPred, y_train)
  if w1.grad is not None: w1.grad.zero_()
  if b1.grad is not None: b1.grad.zero_()
  if w2.grad is not None: w2.grad.zero_()
  if b2.grad is not None: b2.grad.zero_()
  loss.backward()
  w1 = w1.detach() - torch.mul(learning_rate, w1.grad.detach())
  b1 = b1.detach() - torch.mul(learning_rate, b1.grad.detach())
  w2 = w2.detach() - torch.mul(learning_rate, w2.grad.detach())
  b2 = b2.detach() - torch.mul(learning_rate, b2.grad.detach())
  w1.requires_grad = True
  b1.requires_grad = True
  w2.requires_grad = True
  b2.requires_grad = True
  print(loss.item())