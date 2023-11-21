// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

#define TILE_SIZE 16

namespace Flow
{
    __global__
    void MM_TiledKernel( float* arr1, float* arr2, int arr1Rows, int arr1Cols, int arr2Cols, float* result )
    {
        __shared__ float arr1_s[TILE_SIZE][TILE_SIZE];
        __shared__ float arr2_s[TILE_SIZE][TILE_SIZE];
        int threadX = threadIdx.x;
        int threadY = threadIdx.y;
        int row = blockIdx.y * TILE_SIZE + threadY;
        int col = blockIdx.x * TILE_SIZE + threadX;
        float sum = 0.0f;
        for ( int i = 0; i < ceil( arr1Cols / float(TILE_SIZE) ); i++ )
        {
            if ( row < arr1Rows && ( i * TILE_SIZE + threadX ) < arr1Cols )
                arr1_s[threadY][threadX] = arr1[ row * arr1Cols + i * TILE_SIZE + threadX ];
            else arr1_s[threadY][threadX] = 0.0f;
            if ( ( i * TILE_SIZE + threadY ) < arr1Cols && col < arr2Cols )
                arr2_s[threadY][threadX] = arr2[ ( i * TILE_SIZE + threadY ) * arr2Cols + col ];
            else arr2_s[threadY][threadX] = 0.0f;
            __syncthreads();
            for ( int k = 0; k < TILE_SIZE; k++ )
                sum += arr1_s[threadY][k] * arr2_s[k][threadX];
            __syncthreads();
        }
        if ( row < arr1Rows && col < arr2Cols )
            result[ row * arr2Cols + col ] = sum;
    }


    __host__
    NArrayCore* MM_CUDA(NArrayCore* arr1, NArrayCore* arr2)
    {
        int arr1Rows = arr1->GetShape()[0];
        int arr1Cols = arr1->GetShape()[1];
        int arr2Cols = arr2->GetShape()[1];
        float* arr1_d;
        float* arr2_d;
        float* result_d;
        cudaMalloc( (void**)&arr1_d, arr1Rows * arr1Cols * sizeof(float) );
        cudaMalloc( (void**)&arr2_d, arr1Cols * arr2Cols * sizeof(float) );
        cudaMalloc( (void**)&result_d, arr1Rows * arr2Cols * sizeof(float) );
        cudaMemcpy( arr1_d, arr1->GetData(), arr1Rows * arr1Cols * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arr2_d, arr2->GetData(), arr1Cols * arr2Cols * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemset( result_d, 0, arr1Rows * arr2Cols * sizeof(float) );
        dim3 dimGrid( ceil( arr2Cols / float(TILE_SIZE) ), ceil( arr1Rows / float(TILE_SIZE) ), 1 );
        dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1 );
        MM_TiledKernel<<< dimGrid, dimBlock >>>( arr1_d, arr2_d, arr1Rows, arr1Cols, arr2Cols, result_d );
        vector<float> resultData( arr1Rows * arr2Cols );
        cudaMemcpy( resultData.data(), result_d, arr1Rows * arr2Cols * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
        return new NArrayCore( { arr1Rows, arr2Cols }, resultData, { arr1, arr2 }, NArrayCore::Operation::MM );
    }

}

__global__
void BackwardMM_Kernel_1( float* operandGradient1, float* gradient, float* operand2, int arr1Cols, int arr2Cols )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    atomicAdd( &operandGradient1[ i * arr1Cols + j ], gradient[ i * arr2Cols + k ] * operand2[ j * arr2Cols + k ] );
}

__global__
void BackwardMM_Kernel_2( float* operandGradient2, float* gradient, float* operand1, int arr1Rows, int arr1Cols, int arr2Cols )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    atomicAdd( &operandGradient2[ i * arr2Cols + j ], gradient[ k * arr2Cols + j ] * operand1[ k * arr1Cols + i ] );
}

__host__
void Flow::NArrayCore::BackwardMM_CUDA()
{
    int arr1Rows = Operands[0]->GetShape()[0];
    int arr1Cols = Operands[0]->GetShape()[1];
    int arr2Cols = Operands[1]->GetShape()[1];
    float* operand1_d;
    float* operand2_d;
    float* gradient_d;
    float* operandGradient1_d;
    float* operandGradient2_d;
    cudaMalloc( &operand1_d, arr1Rows * arr1Cols * sizeof(float) );
    cudaMalloc( &operand2_d, arr1Cols * arr2Cols * sizeof(float) );
    cudaMalloc( &gradient_d, arr1Rows * arr2Cols * sizeof(float) );
    cudaMalloc( &operandGradient1_d, arr1Rows * arr1Cols * sizeof(float) );
    cudaMalloc( &operandGradient2_d, arr1Cols * arr2Cols * sizeof(float) );
    cudaMemcpy( operand1_d, Operands[0]->GetData(), arr1Rows * arr1Cols * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( operand2_d, Operands[1]->GetData(), arr1Cols * arr2Cols * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( gradient_d, Gradient->GetData(), arr1Rows * arr2Cols * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemset( operandGradient1_d, 0, arr1Rows * arr1Cols * sizeof(float) );
    cudaMemset( operandGradient2_d, 0, arr1Cols * arr2Cols * sizeof(float) );
    dim3 gridDimA( arr1Rows, arr1Cols, arr2Cols );
    dim3 gridDimB( arr1Cols, arr2Cols, arr1Rows );
    BackwardMM_Kernel_1<<< gridDimA, 1 >>>( operandGradient1_d, gradient_d, operand2_d, arr1Cols, arr2Cols );
    BackwardMM_Kernel_2<<< gridDimB, 1 >>>( operandGradient2_d, gradient_d, operand1_d, arr1Rows, arr1Cols, arr2Cols );
    cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient1_d, arr1Rows * arr1Cols * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( Operands[1]->Gradient->Data.data(), operandGradient2_d, arr1Cols * arr2Cols * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(operand1_d);
    cudaFree(operand2_d);
    cudaFree(gradient_d);
    cudaFree(operandGradient1_d);
    cudaFree(operandGradient2_d);
}