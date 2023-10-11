// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Exp_Kernel()
{
    int i = blockIdx.x;

}

namespace Flow
{
    __host__
    void Exp_CUDA()
    {

    }
}

__global__
void BackwardExp_Kernel()
{
    int i = blockIdx.x;

}

__host__
void Flow::NArrayCore::BackwardExp_CUDA()
{
    
}