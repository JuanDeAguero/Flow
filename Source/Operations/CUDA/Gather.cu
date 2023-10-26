// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Gather_Kernel()
{

}

namespace Flow
{
    __host__
    NArrayCore* Gather_CUDA( NArrayCore* arr, int dim, NArrayCore* index )
    {
        return nullptr;
    }
}

__global__
void BackwardGather_Kernel()
{

}

__host__
void Flow::NArrayCore::BackwardGather_CUDA()
{
    
}