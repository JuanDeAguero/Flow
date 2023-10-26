// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Gather_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* index, int* indexShape, int indexShapeSize, float* result )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
        multiIndex[dim] = static_cast<int>(index[i]);
        int flatIndex = MultiToFlatIndex_Device( multiIndex, arrShape, arrShapeSize );
        result[i] = arr[flatIndex];
    }

    __host__
    NArrayCore* Gather_CUDA( NArrayCore* arr, int dim, NArrayCore* index )
    {
        int n = index->Get().size();
        float* arr_d;
        int* arrShape_d;
        float* index_d;
        int* indexShape_d;
        float* result_d;
        cudaMalloc( (void**)&arr_d, arr->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&index_d, index->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&indexShape_d, index->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( arr_d, arr->GetData(), arr->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( index_d, index->GetData(), index->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( indexShape_d, index->GetShapeData(), index->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        Gather_Kernel<<< n, 1 >>>( arr_d, arrShape_d, arr->GetShape().size(), dim, index_d, indexShape_d, index->GetShape().size(), result_d );
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr_d);
        cudaFree(arrShape_d);
        cudaFree(index_d);
        cudaFree(result_d);
        cudaFree(result_d);
        NArrayCore* result = new NArrayCore( index->GetShape(), resultData, { arr }, NArrayCore::Operation::GATHER );
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
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