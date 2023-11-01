import torch

state = torch.load("llama.pth")
for key in state:
  print(key, state[key])