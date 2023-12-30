// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void ReLU_Kernel( float* result )
{
    int i = blockIdx.x;
    Flow::AtomicMax_Device( &result[i], 0.0f );
}

Flow::NArrayCore* Flow::ReLU( NArrayCore* arr )
{
    int n = SizeFromShape(arr->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice );
    ReLU_Kernel<<< n, 1 >>>(result_d);
    cudaDeviceSynchronize();
    return new NArrayCore( arr->GetShape(), result_d, { arr }, NArrayCore::Operation::RELU );
}

__global__
void BackwardReLU_Kernel( float* gradient, float* operand, float* operandGradient )
{
    int i = blockIdx.x;
    float grad = ( operand[i] > 0.0f ) ? gradient[i] : 0.0f;
    atomicAdd( &operandGradient[i], grad );
}

void Flow::NArrayCore::BackwardReLU()
{
    int n = SizeFromShape(Shape);
    BackwardReLU_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(), Operands[0]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}