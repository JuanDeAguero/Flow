// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Mul_Kernel( float* arr1, float* arr2, float* result )
{
    int i = blockIdx.x;
    result[i] = arr1[i] * arr2[i];
}
    
namespace Flow
{
    __host__
    NArrayCore* Mul_CUDA( NArrayCore* arr1, NArrayCore* arr2 )
    {
        int n = arr1->Get().size();
        float* arr1_d;
        float* arr2_d;
        float* result_d;
        cudaMalloc( (void**)&arr1_d, n * sizeof(float) );
        cudaMalloc( (void**)&arr2_d, n * sizeof(float) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( arr1_d, arr1->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arr2_d, arr2->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        Mul_Kernel<<< n, 1 >>>( arr1_d, arr2_d, result_d );
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
        return new NArrayCore( arr1->GetShape(), resultData, { arr1, arr2 }, NArrayCore::Operation::MUL );
    }
}

__global__
void BackwardMul_Kernel( float* gradient, float* operand1, float* operandGradient1, float* operand2, float* operandGradient2 )
{
    int i = blockIdx.x;
    operandGradient1[i] += operand2[i] * gradient[i];
    operandGradient2[i] += operand1[i] * gradient[i];
}

__host__
void Flow::NArrayCore::BackwardMul_CUDA()
{
    int n = Gradient->Data.size();
    float* gradient_d;
    float* operand1_d;
    float* operandGradient1_d;
    float* operand2_d;
    float* operandGradient2_d;
    cudaMalloc( (void**)&gradient_d, n * sizeof(float) );
    cudaMalloc( (void**)&operandGradient1_d, n * sizeof(float) );
    cudaMalloc( (void**)&operand1_d, n * sizeof(float) );
    cudaMalloc( (void**)&operandGradient2_d, n * sizeof(float) );
    cudaMalloc( (void**)&operand2_d, n * sizeof(float) );
    cudaMemcpy( gradient_d, Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operandGradient1_d, Operands[0]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operand1_d, Operands[0]->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operandGradient2_d, Operands[1]->Gradient->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operand2_d, Operands[1]->Data.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    BackwardMul_Kernel<<< n, 1 >>>( gradient_d, operand1_d, operandGradient1_d, operand2_d, operandGradient2_d );
    cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient1_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[1]->Gradient->Data.data(), operandGradient2_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(gradient_d);
    cudaFree(operand1_d);
    cudaFree(operandGradient1_d);
    cudaFree(operand2_d);
    cudaFree(operandGradient2_d);
}