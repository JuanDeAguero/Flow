// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Pow_Kernel( float* result, float exponent )
{
    int i = blockIdx.x;
    result[i] = pow( result[i], exponent );
}

NARRAY Flow::Pow( NARRAY arr, float exponent )
{
    int n = SizeFromShape(arr->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice );
    Pow_Kernel<<< n, 1 >>>( result_d, exponent );
    cudaDeviceSynchronize();
    NARRAY result = Create( arr->GetShape(), result_d, { arr }, NArray::Operation::POW );
    result->Exponent = exponent;
    return result;
}

__global__
void BackwardPow_Kernel( float* gradient, float* operand, float* operandGradient, float exponent )
{
    int i = blockIdx.x;
    float grad = gradient[i] * exponent * pow( operand[i], exponent - 1 );
    atomicAdd( &operandGradient[i], grad );
}

void Flow::NArray::BackwardPow()
{
    int n = SizeFromShape(Shape);
    BackwardPow_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(),
        Operands[0]->GetGradient()->GetData(), Exponent );
    cudaDeviceSynchronize();
}