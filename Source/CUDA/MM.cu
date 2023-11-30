// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

#define TILE_SIZE 16

namespace Flow
{
    __global__
    void MM_TiledKernel( float* arr1, float* arr2, float* result, int arr1Rows, int arr1Cols, int arr2Cols )
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
    NArrayCore* MM_CUDA( NArrayCore* arr1, NArrayCore* arr2 )
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
        MM_TiledKernel<<< dimGrid, dimBlock >>>( arr1_d, arr2_d, result_d, arr1Rows, arr1Cols, arr2Cols );
        vector<float> resultData( arr1Rows * arr2Cols );
        cudaMemcpy( resultData.data(), result_d, arr1Rows * arr2Cols * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr1_d);
        cudaFree(arr2_d);
        cudaFree(result_d);
        return new NArrayCore( { arr1Rows, arr2Cols }, resultData, { arr1, arr2 }, NArrayCore::Operation::MM );
    }
}