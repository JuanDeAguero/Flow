// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Exp_Kernel( float* result )
{
    int i = blockIdx.x;
    result[i] = exp(result[i]);
}

NARRAY Flow::Exp( NARRAY arr )
{
    int n = SizeFromShape(arr->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice );
    Exp_Kernel<<< n, 1 >>>(result_d);
    return Create( arr->GetShape(), result_d, { arr }, NArray::Operation::EXP );
}

__global__
void BackwardExp_Kernel( float* gradient, float* operand, float* operandGradient )
{
    int i = blockIdx.x;
    float grad = gradient[i] * exp(operand[i]);
    atomicAdd( &operandGradient[i], grad );
}

void Flow::NArray::BackwardExp()
{
    int n = SizeFromShape(Shape);
    BackwardExp_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(),
        Operands[0]->GetGradient()->GetData() );
}