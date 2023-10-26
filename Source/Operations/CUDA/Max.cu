// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Max_Kernel()
{

}

namespace Flow
{
    __host__
    NArrayCore* Max_CUDA( NArrayCore* arr, int dim )
    {
        return nullptr;
    }
}

__global__
void BackwardMax_Kernel()
{

}

__host__
void Flow::NArrayCore::BackwardMax_CUDA()
{
    
}