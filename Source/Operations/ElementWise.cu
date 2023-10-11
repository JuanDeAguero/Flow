// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void ElementWise_Kernel( float* arr1, float* arr2, float* result, int op )
{
    int i = blockIdx.x;
    switch (op)
    {
        case 1: result[i] = arr1[i] + arr2[i]; break;
        case 2: result[i] = arr1[i] * arr2[i]; break;
    }
}
    
namespace Flow
{
    __host__
    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        int n = arr1->Get().size();
        float* arr1_d;
        float* arr2_d;
        float* result_d;
        cudaMalloc( (void**)&arr1_d, n * sizeof(float) );
        cudaMalloc( (void**)&arr2_d, n * sizeof(float) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( arr1_d, arr1->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arr2_d, arr2->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        ElementWise_Kernel<<< n, 1 >>>( arr1_d, arr2_d, result_d, static_cast<int>(op) );
        cudaMemcpy( result->GetData(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
    }
}