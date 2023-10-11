// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void ElementWise_Kernel( float* arr1, float* arr2, float* result, int* shape, int shapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
        int index = MultiToFlatIndex_Device( multiIndex, shape, shapeSize );
        result[index] = arr1[index] + arr2[index];
    }
    
    __host__
    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        int n = arr1->Get().size();
        float* arr1_d;
        float* arr2_d;
        float* result_d;
        int* shape_d;
        cudaMalloc( (void**)&arr1_d, n * sizeof(float) );
        cudaMalloc( (void**)&arr2_d, n * sizeof(float) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMalloc( (void**)&shape_d, arr1->GetShape().size() * sizeof(int) );
        cudaMemcpy( arr1_d, arr1->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arr2_d, arr2->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( shape_d, arr1->GetShapeData(), arr1->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        ElementWise_Kernel<<< n, 1 >>>( arr1_d, arr2_d, result_d, shape_d, arr1->GetShape().size() );
        cudaMemcpy( result->GetData(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
    }
}