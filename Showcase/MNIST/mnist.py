import tensorflow.keras as keras
import torch

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train[:100, :]
y_train = y_train[:100]
x_test = x_test[:100, :]
y_test = y_test[:100]

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

x_train = torch.transpose(x_train, 0, 1)
x_test = torch.transpose(x_test, 0, 1)

for i in range(28):
  for j in range(28):
    print(f"{x_train[i * 28 + j][95]:3}", end='')
  print()
print(y_train[95])