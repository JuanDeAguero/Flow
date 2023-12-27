// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Transpose_Kernel( float* arr, int* arrShape, int arrShapeSize, int firstDim, int secondDim, float* result, int* resultShape, int resultShapeSize )
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

std::pair< std::vector<int>, float* > Flow::TransposeRaw( NArrayCore* arr, int firstDim, int secondDim )
{
    int n = SizeFromShape(arr->GetShape());
    std::vector<int> resultShape = arr->GetShape();
    int temp = resultShape[firstDim];
    resultShape[firstDim] = resultShape[secondDim];
    resultShape[secondDim] = temp;
    int* arrShape_d;
    float* result_d;
    int* resultShape_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );
    cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int), cudaMemcpyHostToDevice );
    Transpose_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), firstDim, secondDim, result_d, resultShape_d, resultShape.size() );
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(resultShape_d);
    return { resultShape, result_d };
}

Flow::NArrayCore* Flow::Transpose( NArrayCore* arr, int firstDim, int secondDim )
{
    auto transpose = TransposeRaw( arr, firstDim, secondDim );
    return new NArrayCore( transpose.first, transpose.second, { arr }, NArrayCore::Operation::TRANSPOSE );
}

void Flow::NArrayCore::BackwardTranspose()
{
    throw runtime_error("Backward transpose not implemented!");
}