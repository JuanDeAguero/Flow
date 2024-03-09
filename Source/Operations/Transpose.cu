// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

using namespace std;

__global__
void Transpose_Kernel( float* arr, int* arrShape, int arrShapeSize, int firstDim, int secondDim,
    float* result, int* resultShape, int resultShapeSize )
{
    int i = blockIdx.x;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
    int temp = multiIndex[firstDim];
    multiIndex[firstDim] = multiIndex[secondDim];
    multiIndex[secondDim] = temp;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, resultShape, resultShapeSize );
    result[flatIndex] = arr[i];
}

pair< vector<int>, float* > Flow::TransposeRaw( NARRAY arr, int firstDim, int secondDim )
{
    int n = SizeFromShape(arr->GetShape());
    vector<int> resultShape = arr->GetShape();
    int temp = resultShape[firstDim];
    resultShape[firstDim] = resultShape[secondDim];
    resultShape[secondDim] = temp;
    int* arrShape_d;
    float* result_d;
    int* resultShape_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );
    cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int),
        cudaMemcpyHostToDevice );
    Transpose_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), firstDim,
        secondDim, result_d, resultShape_d, resultShape.size() );
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    return { resultShape, result_d };
}

NARRAY Flow::Transpose( NARRAY arr, int firstDim, int secondDim )
{
    auto transpose = TransposeRaw( arr, firstDim, secondDim );
    NARRAY result = Create( transpose.first, transpose.second, { arr },
        NArray::Operation::TRANSPOSE );
    result->TransposeFirstDim = firstDim;
    result->TransposeSecondDim = secondDim;
    return result;
}

__global__
void BackwardTranspose_Kernel( int* arrShape, int arrShapeSize, float* arrGradient,
    int* operandShape, int operandShapeSize, float* operandGradient, int firstDim, int secondDim )
{
    int i = blockIdx.x;
    int arrSize = 1;
    for ( int i = 0; i < arrShapeSize; i++ ) arrSize *= arrShape[i];
    if ( i >= arrSize ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
    int temp = multiIndex[firstDim];
    multiIndex[firstDim] = multiIndex[secondDim];
    multiIndex[secondDim] = temp;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
    atomicAdd( &operandGradient[flatIndex], arrGradient[i] );
}

void Flow::NArray::BackwardTranspose()
{
    int n = SizeFromShape(GetShape());
    int* arrShape_d;
    int* operandShape_d;
    cudaMalloc( &arrShape_d, GetShape().size() * sizeof(int) );
    cudaMalloc( &operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMemcpy( arrShape_d, GetShape().data(), GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShape().data(),
        Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    BackwardTranspose_Kernel<<< n, 1 >>>( arrShape_d, GetShape().size(), Gradient->GetData(),
        operandShape_d, Operands[0]->GetShape().size(), Operands[0]->GetGradient()->GetData(),
        TransposeFirstDim, TransposeSecondDim );
    cudaFree(arrShape_d);
    cudaFree(operandShape_d);
}