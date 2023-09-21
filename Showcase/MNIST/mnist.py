import numpy as np
import tensorflow.keras as keras
import torch

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

#x_train = x_train[:100, :]
#y_train = y_train[:100]
#x_test = x_test[:100, :]
#y_test = y_test[:100]

x_train = x_train / 255
x_test = x_test / 255

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

for i in range(28):
  for j in range(28):
    if x_train[7][i * 28 + j] > 0:
      print("00", end='')
    else:
      print("..", end='')
  print()
print(y_train[7])

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

def softmax(logits):
  max_val = torch.max(logits, dim=1, keepdim=True)[0]
  return torch.sub(logits, torch.sub(max_val, torch.log(torch.sum(torch.exp(torch.sub(logits, max_val)), dim=1, keepdim=True))))

def cross_entropy(logits, targets):
  return torch.neg(torch.mean(torch.gather(softmax(logits), 1, torch.unsqueeze(targets, -1))))

input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 50

W1 = torch.randn(input_size, hidden_size, requires_grad=True, dtype=torch.float32)
b1 = torch.randn(hidden_size, requires_grad=True, dtype=torch.float32)
W2 = torch.randn(hidden_size, output_size, requires_grad=True, dtype=torch.float32)
b2 = torch.randn(output_size, requires_grad=True, dtype=torch.float32)

for epoch in range(epochs):
    z1 = x_train.mm(W1) + b1
    a1 = torch.relu(z1)
    z2 = a1.mm(W2) + b2
    logits = z2
    loss = cross_entropy(logits, y_train)
    loss.backward()
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.8f}")

with torch.no_grad():
    z1 = x_test.mm(W1) + b1
    a1 = torch.relu(z1)
    z2 = a1.mm(W2) + b2
    logits = z2
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y_test).float().mean().item()

print(f"Test Accuracy: {accuracy*100:.2f}%")