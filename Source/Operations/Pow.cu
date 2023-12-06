// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Pow_Kernel( float* result, float exponent )
{
    int i = blockIdx.x;
    result[i] = pow( result[i], exponent );
}

namespace Flow
{
    __host__
    NArrayCore* Pow( NArrayCore* arr, float exponent )
    {
        return nullptr;
        
        /*int n = arr->Get().size();
        float* result_d;
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        Pow_Kernel<<< n, 1 >>>( result_d, exponent );
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(result_d);
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::POW );
        result->Exponent = exponent;
        return result;*/
    }
}

__global__
void BackwardPow_Kernel( float* gradient, float* operand, float* operandGradient, float exponent )
{
    int i = blockIdx.x;
    float grad = gradient[i] * exponent * pow( operand[i], exponent - 1);
    operandGradient[i] += grad;
}

__host__
void Flow::NArrayCore::BackwardPow()
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
    BackwardPow_Kernel<<< n, 1 >>>( gradient_d, operand_d, operandGradient_d, Exponent );
    cudaMemcpy( Operands[0]->Gradient->GetData(), operandGradient_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(gradient_d);
    cudaFree(operand_d);
    cudaFree(operandGradient_d);*/
}