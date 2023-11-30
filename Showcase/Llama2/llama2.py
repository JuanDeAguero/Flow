# Copyright (c) 2023 Juan M. G. de Agüero

import torch

state = torch.load("llama2.pth")
for key in state:
  print(key, state[key])