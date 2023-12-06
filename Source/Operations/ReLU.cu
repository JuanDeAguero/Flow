// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void ReLU_Kernel( float* result )
{
    int i = blockIdx.x;
    Flow::AtomicMax_Device( &result[i], 0.0f );
}

namespace Flow
{
    __host__
    NArrayCore* ReLU( NArrayCore* arr )
    {
        return nullptr;
        
        /*int n = arr->Get().size();
        float* result_d;
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        ReLU_Kernel<<< n, 1 >>>(result_d);
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(result_d);
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::RELU );*/
    }
}

__global__
void BackwardReLU_Kernel( float* gradient, float* operand, float* operandGradient )
{
    int i = blockIdx.x;
    float grad = ( operand[i] > 0.0f ) ? gradient[i] : 0.0f;
    operandGradient[i] += grad;
}

__host__
void Flow::NArrayCore::BackwardReLU()
{
    /*int n = Data.size();
    float* gradient_d;
    float* operand_d;
    float* operandGradient_d;
    cudaMalloc( (void**)&gradient_d, n * sizeof(float) );
    cudaMalloc( (void**)&operand_d, n * sizeof(float) );
    cudaMalloc( (void**)&operandGradient_d, n * sizeof(float) );
    cudaMemcpy( gradient_d, Gradient->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operand_d, Operands[0]->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operandGradient_d, Operands[0]->GetGradient()->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
    BackwardReLU_Kernel<<< n, 1 >>>( gradient_d, operand_d, operandGradient_d );
    cudaMemcpy( Operands[0]->Gradient->GetData(), operandGradient_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(gradient_d);
    cudaFree(operand_d);
    cudaFree(operandGradient_d);*/
}