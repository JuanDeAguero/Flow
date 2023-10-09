// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArrayCore.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string.h>

namespace Flow
{
    __global__ void ElementWise_Kernel( float* arr1, float* arr2, float* result, int size, NArrayCore::Operation op )
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        if ( id < size )
        {
            switch (op)
            {
                case NArrayCore::Operation::ADD:
                    result[id] = arr1[id] + arr2[id];
                    break;
                case NArrayCore::Operation::MUL:
                    result[id] = arr1[id] * arr2[id];
                    break;
            }
        }
    }

    void cudaAssert( cudaError_t status )
    {
        if ( status != cudaSuccess )
        {
            fprintf( stderr, "CUDA Error: %s\n", cudaGetErrorString(status) );
            exit(1);
        }
    }

    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        int size = SizeFromShape(arr1->GetShape());
        if ( size != SizeFromShape(arr2->GetShape()) )
            throw runtime_error("[ElementWise_CUDA] Arrays shapes don't match for element-wise operation.");
        float* d_arr1;
        float* d_arr2;
        float* d_result;
        cudaAssert(cudaMalloc(&d_arr1, size * sizeof(float)));
        cudaAssert(cudaMalloc(&d_arr2, size * sizeof(float)));
        cudaAssert(cudaMalloc(&d_result, size * sizeof(float)));
        cudaAssert(cudaMemcpy(d_arr1, arr1->Get().data(), size * sizeof(float), cudaMemcpyHostToDevice));
        cudaAssert(cudaMemcpy(d_arr2, arr2->Get().data(), size * sizeof(float), cudaMemcpyHostToDevice));
        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        ElementWise_Kernel<<<gridSize, blockSize>>>(d_arr1, d_arr2, d_result, size, op);
        cudaAssert(cudaGetLastError());
        cudaAssert(cudaDeviceSynchronize());
        cudaAssert(cudaMemcpy(result->Get().data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaAssert(cudaFree(d_arr1));
        cudaAssert(cudaFree(d_arr2));
        cudaAssert(cudaFree(d_result));
    }
}