// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void BackwardAdd_Kernel( float* grad, float* gradOp1, float* gradOp2 )
{
    int i = blockIdx.x;
    gradOp1[i] += grad[i];
    gradOp2[i] += grad[i];
}

__host__
void Flow::NArrayCore::BackwardAdd_CUDA()
{
    int n = Gradient->Data.size();
    float* grad_d;
    float* gradOp1_d;
    float* gradOp2_d;
    cudaMalloc( (void**)&grad_d, n * sizeof(float) );
    cudaMalloc( (void**)&gradOp1_d, n * sizeof(float) );
    cudaMalloc( (void**)&gradOp2_d, n * sizeof(float) );
    cudaMemcpy( grad_d, Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( gradOp1_d, Operands[0]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( gradOp2_d, Operands[1]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    BackwardAdd_Kernel<<< n, 1 >>>( grad_d, gradOp1_d, gradOp2_d );
    cudaMemcpy( Gradient->Data.data(), grad_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[0]->Gradient->Data.data(), gradOp1_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[1]->Gradient->Data.data(), gradOp2_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(grad_d);
    cudaFree(gradOp1_d);
    cudaFree(gradOp2_d);
}