// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

#define TILE_SIZE 32

__global__
void MM_Kernel( float* arr1, float* arr2, float* result, int arr1Rows, int arr1Cols, int arr2Cols )
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

Flow::NArrayCore* Flow::MM( NArrayCore* arr1, NArrayCore* arr2 )
{
    int arr1Rows = arr1->GetShape()[0];
    int arr1Cols = arr1->GetShape()[1];
    int arr2Cols = arr2->GetShape()[1];
    float* result_d;
    cudaMalloc( (void**)&result_d, arr1Rows * arr2Cols * sizeof(float) );
    dim3 dimGrid( ceil( arr2Cols / float(TILE_SIZE) ), ceil( arr1Rows / float(TILE_SIZE) ), 1 );
    dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1 );
    MM_Kernel<<< dimGrid, dimBlock >>>( arr1->GetData(), arr2->GetData(), result_d, arr1Rows, arr1Cols, arr2Cols );
    return new NArrayCore( { arr1Rows, arr2Cols }, result_d, { arr1, arr2 }, NArrayCore::Operation::MM );
}

void Flow::NArrayCore::BackwardMM()
{
    NArrayCore* grad1 = Flow::MM( Gradient->Copy(), Transpose( Operands[1], 0, 1 ) );
    NArrayCore* grad2 = Flow::MM( Transpose( Operands[0], 0, 1 ), Gradient->Copy() );
    grad1->Gradient = nullptr;
    grad2->Gradient = nullptr;
    grad1->Operands = {};
    grad2->Operands = {};
    grad1->Op = NArrayCore::Operation::NONE;
    grad2->Op = NArrayCore::Operation::NONE;
    Operands[0]->Gradient = grad1;
    Operands[1]->Gradient = grad2;
}