// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Log_Kernel( float* result )
{
    int i = blockIdx.x;
    result[i] = log(result[i]);
}

NARRAY Flow::Log( NARRAY arr )
{
    int n = SizeFromShape(arr->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice );
    Log_Kernel<<< n, 1 >>>(result_d);
    cudaDeviceSynchronize();
    return Create( arr->GetShape(), result_d, { arr }, NArray::Operation::LOG );
}

__global__
void BackwardLog_Kernel( float* gradient, float* operand, float* operandGradient )
{
    int i = blockIdx.x;
    float grad = gradient[i] / operand[i];
    atomicAdd( &operandGradient[i], grad );
}

void Flow::NArray::BackwardLog()
{
    int n = SizeFromShape(Shape);
    BackwardLog_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(),
        Operands[0]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}