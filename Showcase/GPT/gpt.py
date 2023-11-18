import torch

with open("input.txt", 'r') as file:
  text = file.read()

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)