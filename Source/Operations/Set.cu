// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Set_Kernel( float* arr, int index, float value )
{
    arr[index] = value;
}

void Flow::NArray::Set( vector<int> coordinates, float value )
{
    int index = MultiToFlatIndex( coordinates, Shape );
    if ( index >= 0 && index < SizeFromShape(Shape) )
    {
        Set_Kernel<<< 1, 1 >>>( Data, index, value );
        cudaDeviceSynchronize();
    }
}

__global__
void Reset_Kernel( float* arr, float value )
{
    int i = blockIdx.x;
    arr[i] = value;
}

void Flow::NArray::Reset( float value )
{
    int n = SizeFromShape(Shape);
    Reset_Kernel<<< n, 1 >>>( Data, value );
    cudaDeviceSynchronize();
}