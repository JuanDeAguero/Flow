// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Log_Kernel()
{

}

namespace Flow
{
    __host__
    NArrayCore* Log_CUDA( NArrayCore* arr )
    {
        return nullptr;
    }
}

__global__
void BackwardLog_Kernel()
{

}

__host__
void Flow::NArrayCore::BackwardLog_CUDA()
{
    
}