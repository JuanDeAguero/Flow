// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

NARRAY Flow::Squeeze( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->GetShape();
    resultShape.erase( resultShape.begin() + dim );
    float* result_d;
    cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), SizeFromShape(arr->GetShape()) * sizeof(float),
        cudaMemcpyDeviceToDevice );
    return Create( resultShape, result_d, { arr }, NArray::Operation::SQUEEZE );
}

__global__
void BackwardSqueeze_Kernel( float* gradient, float* operandGradient )
{
    int i = blockIdx.x;
    atomicAdd( &operandGradient[i], gradient[i] );
}

void Flow::NArray::BackwardSqueeze()
{
    int n = SizeFromShape(Shape);
    BackwardSqueeze_Kernel<<< n, 1 >>>( Gradient->GetData(),
        Operands[0]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}