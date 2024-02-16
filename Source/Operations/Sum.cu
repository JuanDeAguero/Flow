// Copyright (c) 2023 Juan M. G. de Agüero

#include <limits>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Sum_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* result,
    int* resultShape, int resultShapeSize )
{
    int i = blockIdx.x;
    int multiIndex[10];
    Flow::FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
    multiIndex[dim] = 0;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, resultShape, resultShapeSize );
    atomicAdd( &result[flatIndex], arr[i] );
}

NARRAY Flow::Sum( NARRAY arr, int dim )
{
    int n = SizeFromShape(arr->GetShape());
    vector<int> resultShape = arr->GetShape();
    resultShape[dim] = 1;
    vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
    int* arrShape_d;
    float* result_d;
    int* resultShape_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( result_d, resultData.data(), SizeFromShape(resultShape) * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int),
        cudaMemcpyHostToDevice );
    Sum_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), dim, result_d,
        resultShape_d, resultShape.size() );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    NARRAY result = NArray::Create( resultShape, result_d, { arr }, NArray::Operation::SUM );
    result->SumDim = dim;
    return result;
}

__global__
void BackwardSum_Kernel( float* arr, int* shape, int shapeSize, float* gradient, float* operand,
    int* operandShape, int operandShapeSize, float* operandGradient, int dim )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int multiIndex[10];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
    multiIndex[dim] = j;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
    atomicAdd( &operandGradient[flatIndex], gradient[i] );
}

void Flow::NArrayCore::BackwardSum()
{
    int n = SizeFromShape(Shape);
    int* shape_d;
    int* operandShape_d;
    cudaMalloc( (void**)&shape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMemcpy( shape_d, GetShapeData(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(),
        Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    int maxDimSize = Operands[0]->GetShape()[SumDim];
    dim3 gridDims( n, maxDimSize );
    BackwardSum_Kernel<<< gridDims, 1 >>>( GetData(), shape_d, Shape.size(), Gradient->GetData(),
        Operands[0]->GetData(), operandShape_d, Operands[0]->GetShape().size(),
        Operands[0]->GetGradient()->GetData(), SumDim );
    cudaDeviceSynchronize();
    cudaFree(shape_d);
    cudaFree(operandShape_d);
}