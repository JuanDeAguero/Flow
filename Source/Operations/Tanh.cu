// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Tanh_Kernel( float* result )
{
    int i = blockIdx.x;
    result[i] = tanh(result[i]);
}

NARRAY Flow::Tanh( NARRAY arr )
{
    int n = SizeFromShape(arr->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice );
    Tanh_Kernel<<< n, 1 >>>(result_d);
    cudaDeviceSynchronize();
    return NArray::Create( arr->GetShape(), result_d, { arr }, NArray::Operation::TANH );
}

__global__
void BackwardTanh_Kernel( float* gradient, float* operand, float* operandGradient )
{
    int i = blockIdx.x;
    float value = tanh(operand[i]);
    float grad = gradient[i] * ( 1 - value * value );
    atomicAdd( &operandGradient[i], grad );
}

void Flow::NArrayCore::BackwardTanh()
{
    int n = SizeFromShape(Shape);
    BackwardTanh_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(),
        Operands[0]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}