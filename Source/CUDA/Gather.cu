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
        cudaFree(indexShape_d);
        cudaFree(result_d);
        NArrayCore* result = new NArrayCore( index->GetShape(), resultData, { arr }, NArrayCore::Operation::GATHER );
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
    }

    __global__
    void BackwardGather_Kernel( int dim, float* index, int* indexShape, int indexShapeSize, int* operandShape, int operandShapeSize, float* operandGradient, float* gradient )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
        int indexElement = static_cast<int>(index[i]);
        multiIndex[dim] = indexElement;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
        operandGradient[flatIndex] += gradient[i];
    }

    __host__
    void NArrayCore::BackwardGather_CUDA()
    {
        int n = GatherIndex->Data.size();
        float* index_d;
        int* indexShape_d;
        int* operandShape_d;
        float* operandGradient_d;
        float* gradient_d;
        cudaMalloc( (void**)&index_d, GatherIndex->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&indexShape_d, GatherIndex->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&operandGradient_d, Operands[0]->GetGradient()->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&gradient_d, Gradient->Get().size() * sizeof(float) );
        cudaMemcpy( index_d, GatherIndex->GetData(), GatherIndex->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( indexShape_d, GatherIndex->GetShape().data(), GatherIndex->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( operandShape_d, Operands[0]->GetShape().data(), Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( operandGradient_d, Operands[0]->GetGradient()->GetData(), Operands[0]->GetGradient()->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( gradient_d, Gradient->GetData(), Gradient->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        BackwardGather_Kernel<<< n, 1 >>>( GatherDim, index_d, indexShape_d, GatherIndex->GetShape().size(), operandShape_d, Operands[0]->GetShape().size(), operandGradient_d, gradient_d );
        cudaMemcpy( Operands[0]->Gradient->GetData(), operandGradient_d, Operands[0]->Gradient->Get().size() * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(index_d);
        cudaFree(indexShape_d);
        cudaFree(operandShape_d);
        cudaFree(operandGradient_d);
        cudaFree(gradient_d);
    }
}