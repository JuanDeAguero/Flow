// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

NARRAY Flow::Reshape( NARRAY arr, vector<int> shape )
{
    float* result_d;
    cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), SizeFromShape(arr->GetShape()) * sizeof(float),
        cudaMemcpyDeviceToDevice );
    return NArray::Create( shape, result_d, { arr }, NArray::Operation::UNSQUEEZE );
}

__global__
void BackwardReshape_Kernel( float* gradient, float* operandGradient )
{
    int i = blockIdx.x;
    atomicAdd( &operandGradient[i], gradient[i] );
}

void Flow::NArrayCore::BackwardReshape()
{
    int n = SizeFromShape(Shape);
    BackwardReshape_Kernel<<< n, 1 >>>( Gradient->GetData(),
        Operands[0]->GetGradient()->GetData() );   
    cudaDeviceSynchronize();
}