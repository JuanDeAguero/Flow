// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void MM_Kernel( float* arr1, float* arr2, float* result, int n, int p )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    atomicAdd( &result[ i * p + j ], arr1[ i * n + k ] * arr2[ k * p + j ] );
}

namespace Flow
{
    __host__
    NArrayCore* MM_CUDA( NArrayCore* arr1, NArrayCore* arr2 )
    {
        int m = arr1->GetShape()[0];
        int n = arr1->GetShape()[1];
        int p = arr2->GetShape()[1];
        float* arr1_d;
        float* arr2_d;
        float* result_d;
        cudaMalloc( (void**)&arr1_d, m * n * sizeof(float) );
        cudaMalloc( (void**)&arr2_d, n * p * sizeof(float) );
        cudaMalloc( (void**)&result_d, m * p * sizeof(float) );
        cudaMemcpy( arr1_d, arr1->GetData(), m * n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arr2_d, arr2->GetData(), n * p * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemset( result_d, 0, m * p * sizeof(float) );
        dim3 gridDim( m, p, n );
        MM_Kernel<<< gridDim, 1 >>>( arr1_d, arr2_d, result_d, n, p );
        vector<float> resultData( m * p );
        cudaMemcpy( resultData.data(), result_d, m * p * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
        return new NArrayCore( { m, p }, resultData, { arr1, arr2 }, NArrayCore::Operation::MM );
    }
}

__global__
void BackwardMM_Kernel_1( float* operandGradient1, float* gradient, float* operand2, int n, int p )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    atomicAdd( &operandGradient1[ i * n + j ], gradient[ i * p + k ] * operand2[ j * p + k ] );
}

__global__
void BackwardMM_Kernel_2( float* operandGradient2, float* gradient, float* operand1, int m, int n, int p )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    atomicAdd( &operandGradient2[ i * p + j ], gradient[ k * p + j ] * operand1[ k * n + i ] );
}

__host__
void Flow::NArrayCore::BackwardMM_CUDA()
{
    int m = Operands[0]->GetShape()[0];
    int n = Operands[0]->GetShape()[1];
    int p = Operands[1]->GetShape()[1];
    float* operand1_d;
    float* operand2_d;
    float* gradient_d;
    float* operandGradient1_d;
    float* operandGradient2_d;
    cudaMalloc( &operand1_d, m * n * sizeof(float) );
    cudaMalloc( &operand2_d, n * p * sizeof(float) );
    cudaMalloc( &gradient_d, m * p * sizeof(float) );
    cudaMalloc( &operandGradient1_d, m * n * sizeof(float) );
    cudaMalloc( &operandGradient2_d, n * p * sizeof(float) );
    cudaMemcpy( operand1_d, Operands[0]->GetData(), m * n * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operand2_d, Operands[1]->GetData(), n * p * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( gradient_d, Gradient->GetData(), m * p * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemset( operandGradient1_d, 0, m * n * sizeof(float) );
    cudaMemset( operandGradient2_d, 0, n * p * sizeof(float) );
    dim3 gridDimA( m, n, p );
    dim3 gridDimB( n, p, m );
    BackwardMM_Kernel_1<<< gridDimA, 1 >>>( operandGradient1_d, gradient_d, operand2_d, n, p );
    BackwardMM_Kernel_2<<< gridDimB, 1 >>>( operandGradient2_d, gradient_d, operand1_d, m, n, p );
    cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient1_d, m * n * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[1]->Gradient->Data.data(), operandGradient2_d, n * p * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(operand1_d);
    cudaFree(operand2_d);
    cudaFree(gradient_d);
    cudaFree(operandGradient1_d);
    cudaFree(operandGradient2_d);
}