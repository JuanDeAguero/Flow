// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Fold2d_Kernel( float* arr, int* arrShape, int arrShapeSize, float* result, int* resultShape,
    int resultShapeSize, int batchSize, int channels, int kernelHeight, int kernelWidth,
    int outHeight, int outWidth )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= outHeight || j >= outWidth ) return;
    for ( int b = 0; b < batchSize; b++ )
    for ( int c = 0; c < channels; c++ )
    for ( int ki = 0; ki < kernelHeight; ki++ )
    for ( int kj = 0; kj < kernelWidth; kj++ )
    {
        int arrMultiIndex[MAX_DIMS];
        arrMultiIndex[0] = b;
        arrMultiIndex[1] = ( c * kernelHeight * kernelWidth ) + ( ki * kernelWidth ) + kj;
        arrMultiIndex[2] = ( i * outWidth ) + j;
        int resultMultiIndex[MAX_DIMS];
        resultMultiIndex[0] = b;
        resultMultiIndex[1] = c;
        resultMultiIndex[2] = i + ki;
        resultMultiIndex[3] = j + kj;
        int arrIndex = Flow::MultiToFlatIndex_Device( arrMultiIndex, arrShape, arrShapeSize );
        int resultIndex = Flow::MultiToFlatIndex_Device( resultMultiIndex, resultShape,
            resultShapeSize );
        atomicAdd( &result[resultIndex], arr[arrIndex] );
    }
}

NARRAY Flow::Fold2d( NARRAY arr, vector<int> outShape, vector<int> kernel )
{
    int batchSize = outShape[0];
    int channels = outShape[1];
    int height = outShape[2];
    int width = outShape[3];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    vector<int> resultShape = { batchSize, channels, height, width };
    int* arrShape_d;
    float* result_d;
    int* resultShape_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, SizeFromShape(resultShape) * sizeof(float) );
    cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int),
        cudaMemcpyHostToDevice );
    cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int),
        cudaMemcpyHostToDevice );
    Reset_Kernel<<< SizeFromShape(resultShape), 1 >>>( result_d, SizeFromShape(resultShape), 0.0f );
    cudaDeviceSynchronize();
    dim3 threadsPerBlock( 16, 16 );
    int blocksX = ( outWidth + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    int blocksY = ( outHeight + threadsPerBlock.y - 1 ) / threadsPerBlock.y;
    dim3 numBlocks( blocksX, blocksY );
    Fold2d_Kernel<<< numBlocks, threadsPerBlock >>>( arr->GetData(), arrShape_d,
        arr->GetShape().size(), result_d, resultShape_d, resultShape.size(), batchSize, channels,
        kernelHeight, kernelWidth, outHeight, outWidth );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    NARRAY result = Create( resultShape, result_d, { arr }, NArray::Operation::FOLD2D );
    result->FoldOutShape2d = outShape;
    result->FoldKernel2d = kernel;
    return result;
}

__global__
void BackwardFold2d_Kernel( int* arrShape, int arrShapeSize, float* arrGradient, int* operandShape,
    int operandShapeSize, float* operandGradient, int batchSize, int channels, int kernelHeight,
    int kernelWidth, int outHeight, int outWidth )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= outHeight || j >= outWidth ) return;
    for ( int b = 0; b < batchSize; b++ )
    for ( int c = 0; c < channels; c++ )
    for ( int ki = 0; ki < kernelHeight; ki++ )
    for ( int kj = 0; kj < kernelWidth; kj++ )
    {
        int arrMultiIndex[MAX_DIMS];
        arrMultiIndex[0] = b;
        arrMultiIndex[1] = c;
        arrMultiIndex[2] = i + ki;
        arrMultiIndex[3] = j + kj;
        int operandMultiIndex[MAX_DIMS];
        operandMultiIndex[0] = b;
        operandMultiIndex[1] = ( c * kernelHeight * kernelWidth ) + ( ki * kernelWidth ) + kj;
        operandMultiIndex[2] = ( i * outWidth ) + j;
        int arrIndex = Flow::MultiToFlatIndex_Device( arrMultiIndex, arrShape, arrShapeSize );
        int operandIndex = Flow::MultiToFlatIndex_Device( operandMultiIndex, operandShape,
            operandShapeSize );
        atomicAdd( &operandGradient[operandIndex], arrGradient[arrIndex] );
    }
}

void Flow::NArray::BackwardFold2d()
{
    int batchSize = FoldOutShape2d[0];
    int channels = FoldOutShape2d[1];
    int height = FoldOutShape2d[2];
    int width = FoldOutShape2d[3];
    int kernelHeight = FoldKernel2d[0];
    int kernelWidth = FoldKernel2d[1];
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    int* arrShape_d;
    int* operandShape_d;
    cudaMalloc( (void**)&arrShape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->GetShape().size() * sizeof(int) );
    cudaMemcpy( arrShape_d, Shape.data(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(),
        Operands[0]->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    dim3 threadsPerBlock( 16, 16 );
    int blocksX = ( outWidth + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    int blocksY = ( outHeight + threadsPerBlock.y - 1 ) / threadsPerBlock.y;
    dim3 numBlocks( blocksX, blocksY );
    BackwardFold2d_Kernel<<< numBlocks, threadsPerBlock >>>( arrShape_d, Shape.size(),
        Gradient->GetData(), operandShape_d, Operands[0]->GetShape().size(),
        Operands[0]->GetGradient()->GetData(), batchSize, channels, kernelHeight, kernelWidth,
        outHeight, outWidth );
    cudaDeviceSynchronize();
}