// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"
/*
__global__
void Index_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* index,
    float* result, int* resultShape, int resultShapeSize )
{
    int i = blockIdx.x;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, resultShape, resultShapeSize, multiIndex );
    multiIndex[dim] = index[multiIndex[dim]];
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, arrShape, arrShapeSize );
    result[i] = arr[flatIndex];
}

NARRAY Flow::Index( NARRAY arr, int dim, NARRAY index )
{
    vector<int> resultShape = arr->GetShape();
    resultShape[dim] = SizeFromShape(index->GetShape());
    int n = SizeFromShape(resultShape);
    int* arrShape_d;
    float* result_d;
    int* resultShape_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int),
        cudaMemcpyHostToDevice );
    Index_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), dim,
        index->GetData(), result_d, resultShape_d, resultShape.size() );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    NARRAY result = Create( resultShape, result_d, { arr, index }, NArray::Operation::INDEX );
    result->IndexDim = dim;
    result->Index = index;
    return result;
}

__global__
void BackwardIndex_Kernel( int* shape, int shapeSize, float* gradient, int* operandShape,
    int operandShapeSize, float* operandGradient, int dim, float* index )
{
    int i = blockIdx.x;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
    multiIndex[dim] = index[multiIndex[dim]];
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
    atomicAdd( &operandGradient[flatIndex], gradient[i] );
}

void Flow::NArray::BackwardIndex()
{
    int n = SizeFromShape(Gradient->GetShape());
    int* shape_d;
    int* operandShape_d;
    cudaMalloc( (void**)&shape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMemcpy( shape_d, GetShapeData(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(),
        Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    BackwardIndex_Kernel<<< n, 1 >>>( shape_d, Shape.size(), Gradient->GetData(), operandShape_d,
        Operands[0]->GetShape().size(), Operands[0]->GetGradient()->GetData(), IndexDim,
        Index->GetData() );
    cudaDeviceSynchronize();
    cudaFree(shape_d);
    cudaFree(operandShape_d);
}*/