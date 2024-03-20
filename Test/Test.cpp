// Copyright (c) Juan M. G. de Agüero 2023

#include <torch/torch.h>

#include "Flow.h"

using namespace std;

int main()
{
    torch::Tensor tensor = torch::rand({ 2, 3 });
    Flow::Print(tensor[0][0].item<float>());
}