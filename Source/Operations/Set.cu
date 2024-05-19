// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Set_Kernel( float* arr, float value, int index )
{
    arr[index] = value;
}

void Flow::NArray::Set( vector<int> coordinates, float value )
{
    int index = MultiToFlatIndex( coordinates, Stride, StorageOffset );
    if ( index >= 0 && index < SizeFromShape(Shape) )
    {
        Set_Kernel<<< 1, 1 >>>( Data, value, index );
        CUDA_DeviceSynchronize();
    }
}

__global__
void Flow::Reset_Kernel( float* arr, float value, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    arr[i] = value;
}

void Flow::NArray::Reset( float value )
{
    int n = SizeFromShape(Shape);
    Reset_Kernel<<< BLOCKS(n), TPB >>>( Data, value, n );
    CUDA_DeviceSynchronize();
}