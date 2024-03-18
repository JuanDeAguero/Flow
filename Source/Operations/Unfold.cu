// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Unfold2d_Kernel( float* arr, int* arrShape, int arrShapeSize, float* result, int* resultShape,
    int resultShapeSize, int batchSize, int channels, int kernelHeight, int kernelWidth,
    int strideHeight, int strideWidth, int outHeight, int outWidth )
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
        arrMultiIndex[2] = ( i * strideHeight ) + ki;
        arrMultiIndex[3] = ( j * strideWidth ) + kj;
        int resultMultiIndex[MAX_DIMS];
        resultMultiIndex[0] = b;
        resultMultiIndex[1] = ( c * kernelHeight * kernelWidth ) + ( ki * kernelWidth ) + kj;
        resultMultiIndex[2] = ( i * outWidth ) + j;
        int arrIndex = Flow::MultiToFlatIndex_Device( arrMultiIndex, arrShape, arrShapeSize );
        int resultIndex = Flow::MultiToFlatIndex_Device( resultMultiIndex, resultShape,
            resultShapeSize );
        result[resultIndex] = arr[arrIndex];
    }
}

NARRAY Flow::Unfold2d( NARRAY arr, vector<int> kernel, vector<int> stride )
{
    int batchSize = arr->GetShape()[0];
    int channels = arr->GetShape()[1];
    int height = arr->GetShape()[2];
    int width = arr->GetShape()[3];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int strideHeight = stride[0];
    int strideWidth = stride[1];
    int outHeight = ( ( height - kernelHeight ) / strideHeight ) + 1;
    int outWidth = ( ( width - kernelWidth ) / strideWidth ) + 1;
    vector<int> resultShape = { batchSize, channels * kernelHeight * kernelWidth,
        outHeight * outWidth };
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
    dim3 threadsPerBlock( 16, 16 );
    int blocksX = ( outWidth + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    int blocksY = ( outHeight + threadsPerBlock.y - 1 ) / threadsPerBlock.y;
    dim3 numBlocks( blocksX, blocksY );
    Unfold2d_Kernel<<< numBlocks, threadsPerBlock >>>( arr->GetData(), arrShape_d,
        arr->GetShape().size(), result_d, resultShape_d, resultShape.size(), batchSize, channels,
        kernelHeight, kernelWidth, strideHeight, strideWidth, outHeight, outWidth );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    NARRAY result = Create( resultShape, result_d, { arr }, NArray::Operation::UNFOLD2D );
    result->UnfoldKernel2d = kernel;
    result->UnfoldStride2d = stride;
    return result;
}

__global__
void BackwardUnfold2d_Kernel( int* arrShape, int arrShapeSize, float* arrGradient,
    int* operandShape, int operandShapeSize, float* operandGradient, int batchSize, int channels,
    int kernelHeight, int kernelWidth, int strideHeight, int strideWidth, int outHeight,
    int outWidth )
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
        int operandMultiIndex[MAX_DIMS];
        operandMultiIndex[0] = b;
        operandMultiIndex[1] = c;
        operandMultiIndex[2] = ( i * strideHeight ) + ki;
        operandMultiIndex[3] = ( j * strideWidth ) + kj;
        int arrIndex = Flow::MultiToFlatIndex_Device( arrMultiIndex, arrShape, arrShapeSize );
        int operandIndex = Flow::MultiToFlatIndex_Device( operandMultiIndex, operandShape,
            operandShapeSize );
        atomicAdd( &operandGradient[operandIndex], arrGradient[arrIndex] );
    }
}

void Flow::NArray::BackwardUnfold2d()
{
    int batchSize = Operands[0]->GetShape()[0];
    int channels = Operands[0]->GetShape()[1];
    int height = Operands[0]->GetShape()[2];
    int width = Operands[0]->GetShape()[3];
    int kernelHeight = UnfoldKernel2d[0];
    int kernelWidth = UnfoldKernel2d[1];
    int strideHeight = UnfoldStride2d[0];
    int strideWidth = UnfoldStride2d[1];
    int outHeight = ( ( height - kernelHeight ) / strideHeight ) + 1;
    int outWidth = ( ( width - kernelWidth ) / strideWidth ) + 1;
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
    BackwardUnfold2d_Kernel<<< numBlocks, threadsPerBlock >>>( arrShape_d, Shape.size(),
        Gradient->GetData(), operandShape_d, Operands[0]->GetShape().size(),
        Operands[0]->GetGradient()->GetData(), batchSize, channels, kernelHeight, kernelWidth,
        strideHeight, strideWidth, outHeight, outWidth );
    cudaDeviceSynchronize();
}