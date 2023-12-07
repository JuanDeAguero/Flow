// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Gather_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* index, int* indexShape, int indexShapeSize, float* result )
{
    int i = blockIdx.x;
    int multiIndex[10];
    Flow::FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
    multiIndex[dim] = static_cast<int>(index[i]);
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, arrShape, arrShapeSize );
    result[i] = arr[flatIndex];
}

Flow::NArrayCore* Flow::Gather( NArrayCore* arr, int dim, NArrayCore* index )
{
    int n = SizeFromShape(arr->GetShape());
    int* arrShape_d;
    int* indexShape_d;
    float* result_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&indexShape_d, index->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( indexShape_d, index->GetShapeData(), index->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
    Gather_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), dim, index->GetData(), indexShape_d, index->GetShape().size(), result_d );
    NArrayCore* result = new NArrayCore( index->GetShape(), result_d, { arr }, NArrayCore::Operation::GATHER );
    result->GatherDim = dim;
    result->GatherIndex = index;
    return result;
}

__global__
void BackwardGather_Kernel( float* gradient, int* operandShape, int operandShapeSize, float* operandGradient, int dim, float* index, int* indexShape, int indexShapeSize )
{
    int i = blockIdx.x;
    int multiIndex[10];
    Flow::FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
    int indexElement = static_cast<int>(index[i]);
    multiIndex[dim] = indexElement;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
    operandGradient[flatIndex] += gradient[i];
}

void Flow::NArrayCore::BackwardGather()
{
    int n = SizeFromShape(GatherIndex->GetShape());
    int* operandShape_d;
    int* indexShape_d;
    cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&indexShape_d, GatherIndex->GetShape().size() * sizeof(int) );
    cudaMemcpy( operandShape_d, Operands[0]->GetShape().data(), Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( indexShape_d, GatherIndex->GetShape().data(), GatherIndex->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    BackwardGather_Kernel<<< n, 1 >>>( Gradient->GetData(), operandShape_d, Operands[0]->GetShape().size(), Operands[0]->GetGradient()->GetData(), GatherDim, GatherIndex->GetData(), indexShape_d, GatherIndex->GetShape().size() );
}