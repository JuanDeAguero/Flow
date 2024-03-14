// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Gather_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* index,
    int indexSize, int* indexShape, int indexShapeSize, float* result, int resultSize )
{
    int i = blockIdx.x;
    if ( i >= indexSize ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
    if ( index[i] < 0 || index[i] >= arrShape[dim] ) return;
    multiIndex[dim] = (int)index[i];
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, arrShape, arrShapeSize );
    if ( flatIndex < 0 || flatIndex >= resultSize ) return;
    result[i] = arr[flatIndex];
}

NARRAY Flow::Gather( NARRAY arr, int dim, NARRAY index )
{
    int n = SizeFromShape(arr->GetShape());
    int* arrShape_d;
    int* indexShape_d;
    float* result_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&indexShape_d, index->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( indexShape_d, index->GetShapeData(), index->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemset( result_d, n * sizeof(float), 0.0f );
    Gather_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), dim,
        index->GetData(), SizeFromShape(index->GetShape()), indexShape_d, index->GetShape().size(),
        result_d, n );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(indexShape_d);
    NARRAY result = Create( index->GetShape(), result_d, { arr }, NArray::Operation::GATHER );
    result->GatherDim = dim;
    result->GatherIndex = index;
    return result;
}

__global__
void BackwardGather_Kernel( float* gradient, int* operandShape, int operandShapeSize,
    float* operandGradient, int operandGradientSize, int dim, float* index, int indexSize,
    int* indexShape, int indexShapeSize )
{
    int i = blockIdx.x;
    if ( i >= indexSize ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, indexShape, indexShapeSize, multiIndex );
    int indexElement = (int)index[i];
    if ( indexElement < 0 || indexElement >= operandShape[dim] ) return;
    multiIndex[dim] = indexElement;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
    if ( flatIndex < 0 || flatIndex >= operandGradientSize ) return;
    atomicAdd( &operandGradient[flatIndex], gradient[i] );
}

void Flow::NArray::BackwardGather()
{
    int n = SizeFromShape(GatherIndex->GetShape());
    int* operandShape_d;
    int* indexShape_d;
    cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&indexShape_d, GatherIndex->GetShape().size() * sizeof(int) );
    cudaMemcpy( operandShape_d, Operands[0]->GetShape().data(),
        Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( indexShape_d, GatherIndex->GetShape().data(),
        GatherIndex->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    BackwardGather_Kernel<<< n, 1 >>>( Gradient->GetData(), operandShape_d,
        Operands[0]->GetShape().size(), Operands[0]->GetGradient()->GetData(),
        SizeFromShape(Operands[0]->GetGradient()->GetShape()), GatherDim, GatherIndex->GetData(),
        SizeFromShape(GatherIndex->GetShape()), indexShape_d, GatherIndex->GetShape().size() );
    cudaDeviceSynchronize();
    cudaFree(operandShape_d);
    cudaFree(indexShape_d);
}