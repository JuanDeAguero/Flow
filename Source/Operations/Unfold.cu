// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Unfold2d_Kernel( float* arr, int* arrShape, int arrShapeSize, float* result, int* resultShape,
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
        int multiIndexArr[MAX_DIMS];
        multiIndexArr[0] = b;
        multiIndexArr[1] = c;
        multiIndexArr[2] = i + ki;
        multiIndexArr[3] = j + kj;
        int multiIndexResult[MAX_DIMS];
        multiIndexResult[0] = b;
        multiIndexResult[1] = ( c * kernelHeight * kernelWidth ) + ( ki * kernelWidth ) + kj;
        multiIndexResult[2] = ( i * outWidth ) + j;
        int arrIndex = Flow::MultiToFlatIndex_Device( multiIndexArr, arrShape, arrShapeSize );
        int resultIndex = Flow::MultiToFlatIndex_Device( multiIndexResult, resultShape,
            resultShapeSize );
        result[resultIndex] = arr[arrIndex];
    }
}

NARRAY Flow::Unfold2d( NARRAY arr, vector<int> kernel )
{
    int batchSize = arr->GetShape()[0];
    int channels = arr->GetShape()[1];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int outHeight = arr->GetShape()[2] - kernelHeight + 1;
    int outWidth = arr->GetShape()[3] - kernelWidth + 1;
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
        kernelHeight, kernelWidth, outHeight, outWidth );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    NARRAY result = Create( resultShape, result_d, { arr }, NArray::Operation::UNFOLD2D );
    result->UnfoldKernel2d = kernel;
    return result;
}

__global__
void BackwardUnfold2d_Kernel( float* arrGradient, float* operandGradient )
{

}

void Flow::NArray::BackwardUnfold2d()
{

}