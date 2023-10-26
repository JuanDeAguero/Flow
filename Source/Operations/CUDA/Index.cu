// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Index_Kernel()
{

}

namespace Flow
{
    __host__
    NArrayCore* Index_CUDA( NArrayCore* arr, int dim, NArrayCore* index )
    {
        return nullptr;
    }
}

__global__
void BackwardIndex_Kernel()
{

}

__host__
void Flow::NArrayCore::BackwardIndex_CUDA()
{
    
}