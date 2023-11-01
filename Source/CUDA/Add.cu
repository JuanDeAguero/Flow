// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void BackwardAdd_Kernel( float* gradient, float* operandGradient1, float* operandGradient2 )
{
    int i = blockIdx.x;
    operandGradient1[i] += gradient[i];
    operandGradient2[i] += gradient[i];
}

__host__
void Flow::NArrayCore::BackwardAdd_CUDA()
{
    int n = Gradient->Data.size();
    float* gradient_d;
    float* operandGradient1_d;
    float* operandGradient2_d;
    cudaMalloc( (void**)&gradient_d, n * sizeof(float) );
    cudaMalloc( (void**)&operandGradient1_d, n * sizeof(float) );
    cudaMalloc( (void**)&operandGradient2_d, n * sizeof(float) );
    cudaMemcpy( gradient_d, Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operandGradient1_d, Operands[0]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operandGradient2_d, Operands[1]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    BackwardAdd_Kernel<<< n, 1 >>>( gradient_d, operandGradient1_d, operandGradient2_d );
    cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient1_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[1]->Gradient->Data.data(), operandGradient2_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(gradient_d);
    cudaFree(operandGradient1_d);
    cudaFree(operandGradient2_d);
}