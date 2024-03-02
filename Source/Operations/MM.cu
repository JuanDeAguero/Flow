// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

#define TILE_SIZE 32

using namespace std;

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

pair< vector<int>, float* > Flow::MMRaw( pair< vector<int>, float* > arr1,
    pair< vector<int>, float* > arr2 )
{
    int arr1Rows = arr1.first[0];
    int arr1Cols = arr1.first[1];
    int arr2Cols = arr2.first[1];
    float* result_d;
    cudaMalloc( (void**)&result_d, arr1Rows * arr2Cols * sizeof(float) );
    dim3 dimGrid( ceil( arr2Cols / float(TILE_SIZE) ), ceil( arr1Rows / float(TILE_SIZE) ), 1 );
    dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1 );
    MM_Kernel<<< dimGrid, dimBlock >>>( arr1.second, arr2.second, result_d, arr1Rows, arr1Cols,
        arr2Cols );
    cudaDeviceSynchronize();
    return { { arr1Rows, arr2Cols }, result_d };
}

pair< vector<int>, float* > Flow::MMRaw( NARRAY arr1, NARRAY arr2 )
{
    return MMRaw( { arr1->GetShape(), arr1->GetData() }, { arr2->GetShape(), arr2->GetData() } );
}

pair< vector<int>, float* > Flow::MMRaw( pair< vector<int>, float* > arr1, NARRAY arr2 )
{
    return MMRaw( arr1, { arr2->GetShape(), arr2->GetData() } );
}

pair< vector<int>, float* > Flow::MMRaw( NARRAY arr1, pair< vector<int>, float* > arr2 )
{
    return MMRaw( { arr1->GetShape(), arr1->GetData() }, arr2 );
}

NARRAY Flow::MM( NARRAY arr1, NARRAY arr2 )
{
    auto mm = MMRaw( arr1, arr2 );
    return Create( mm.first, mm.second, { arr1, arr2 }, NArray::Operation::MM );
}

void Flow::NArray::BackwardMM()
{
    auto transpose1 = TransposeRaw( Operands[0], 0, 1 );
    auto transpose2 = TransposeRaw( Operands[1], 0, 1 );
    auto grad1 = MMRaw( Gradient, transpose2 );
    auto grad2 = MMRaw( transpose1, Gradient );
    cudaMemcpy( Operands[0]->Gradient->Data, grad1.second,
        SizeFromShape(grad1.first) * sizeof(float), cudaMemcpyDeviceToDevice );
    cudaMemcpy( Operands[1]->Gradient->Data, grad2.second,
        SizeFromShape(grad2.first) * sizeof(float), cudaMemcpyDeviceToDevice );
    cudaFree(grad1.second);
    cudaFree(grad2.second);
    cudaFree(transpose1.second);
    cudaFree(transpose2.second);
}