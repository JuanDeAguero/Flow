// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Unfold2d_Kernel( float* arr, float* result, int batchSize, int channels, int height, int width,
    int kernelHeight, int kernelWidth, int outHeight, int outWidth )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= outWidth || j >= outHeight ) return;
    int patchIndex = j * outWidth + i;
    for ( int b = 0; b < batchSize; b++ )
    for ( int c = 0; c < channels; c++ )
    for ( int ki = 0; ki < kernelHeight; ki++ )
    for ( int kj = 0; kj < kernelWidth; kj++ )
    {
        int arrIndex = ( ( b * channels + c ) * height + j + ki ) * width + i + kj;
        int batchOffset = b * channels * kernelHeight * kernelWidth;
        int channelOffset = c * kernelHeight * kernelWidth;
        int kernelOffset = ki * kernelWidth + kj;
        int flatOffset = batchOffset + channelOffset + kernelOffset;
        int resultIndex = flatOffset * outHeight * outWidth + patchIndex;
        result[resultIndex] = arr[arrIndex];
    }
}

NARRAY Flow::Unfold2d( NARRAY arr, vector<int> kernel )
{
    int batchSize = arr->GetShape()[0];
    int channels = arr->GetShape()[1];
    int height = arr->GetShape()[2];
    int width = arr->GetShape()[3];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    vector<int> resultShape = { batchSize, channels * kernelHeight * kernelWidth,
        outHeight * outWidth };
    float* result_d;
    size_t resultSize = SizeFromShape(resultShape) * sizeof(float);
    cudaMalloc( (void**)&result_d, resultSize );
    dim3 threadsPerBlock( 16, 16 );
    int blocksX = ( outWidth + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    int blocksY = ( outHeight + threadsPerBlock.y - 1 ) / threadsPerBlock.y;
    dim3 numBlocks( blocksX, blocksY );
    Unfold2d_Kernel<<< numBlocks, threadsPerBlock >>>( arr->GetData(), result_d, batchSize,
        channels, height, width, kernelHeight, kernelWidth, outHeight, outWidth );
    NARRAY result = Create( resultShape, result_d, { arr }, NArray::Operation::UNFOLD2D );
    result->UnfoldKernel2d = kernel;
    return result;
}

__global__
void BackwardUnfold2d_Kernel( float* arrGradient, float* operandGradient, int batchSize,
    int channels, int height, int width, int kernelHeight, int kernelWidth, int outHeight,
    int outWidth )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= outWidth || j >= outHeight ) return;
    int patchIndex = j * outWidth + i;
    for ( int b = 0; b < batchSize; b++ )
    for ( int c = 0; c < channels; c++ )
    for ( int ki = 0; ki < kernelHeight; ki++ )
    for ( int kj = 0; kj < kernelWidth; kj++ )
    {
        int arrIndex = ( ( b * channels + c ) * height + j + ki ) * width + i + kj;
        int batchOffset = b * channels * kernelHeight * kernelWidth;
        int channelOffset = c * kernelHeight * kernelWidth;
        int kernelOffset = ki * kernelWidth + kj;
        int flatOffset = batchOffset + channelOffset + kernelOffset;
        int resultIndex = flatOffset * outHeight * outWidth + patchIndex;
        atomicAdd( &operandGradient[arrIndex], arrGradient[resultIndex] );
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
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    dim3 threadsPerBlock( 16, 16 );
    int blocksX = ( outWidth + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    int blocksY = ( outHeight + threadsPerBlock.y - 1 ) / threadsPerBlock.y;
    dim3 numBlocks( blocksX, blocksY );
    BackwardUnfold2d_Kernel<<< numBlocks, threadsPerBlock >>>( Gradient->GetData(),
        Operands[0]->GetGradient()->GetData(), batchSize, channels, height, width, kernelHeight,
        kernelWidth, outHeight, outWidth );
}