// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <cuda_runtime.h>

#include "Flow/NArrayCore.h"

__global__ void VertorAdd( float* v1, float* v2, float* result, int numElements )
{
    int i = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    if ( i < numElements )
        result[i] = v1[i] + v2[i];
}