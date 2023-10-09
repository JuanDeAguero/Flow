// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArrayCore.h"

#include <cuda_runtime.h>

namespace Flow
{// CUDA Kernel
__global__ void ElementWise_Kernel(float* arr1, float* arr2, float* result, int totalSize, NArrayCore::Operation op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < totalSize) {
        switch (op) {
            case NArrayCore::Operation::ADD:
                result[tid] = arr1[tid] + arr2[tid];
                break;
            case NArrayCore::Operation::MUL:
                result[tid] = arr1[tid] * arr2[tid];
                break;
            // Add other operations as needed...
        }
    }
}

// CUDA error checking
void checkCudaErrors(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
}

// CUDA wrapper function
void ElementWise_CUDA(NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op) {
    int totalSize = SizeFromShape(arr1->GetShape());

    if (totalSize != SizeFromShape(arr2->GetShape())) {
        fprintf(stderr, "Arrays shapes don't match for element-wise operations.\n");
        exit(1);
    }

    // Allocate GPU memory
    float* d_arr1;
    float* d_arr2;
    float* d_result;
    checkCudaErrors(cudaMalloc(&d_arr1, totalSize * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_arr2, totalSize * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_result, totalSize * sizeof(float)));

    // Transfer data to GPU
    checkCudaErrors(cudaMemcpy(d_arr1, arr1->Get().data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_arr2, arr2->Get().data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));

    // Define launch configuration
    const int blockSize = 256; // This can be optimized for specific hardware
    const int gridSize = (totalSize + blockSize - 1) / blockSize;

    // Launch the kernel
    ElementWise_Kernel<<<gridSize, blockSize>>>(d_arr1, d_arr2, d_result, totalSize, op);
    checkCudaErrors(cudaGetLastError()); // Check for errors in kernel launch

    // Synchronize threads to ensure all have completed
    checkCudaErrors(cudaDeviceSynchronize());

    // Transfer result back to host
    checkCudaErrors(cudaMemcpy(result->Get().data(), d_result, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up GPU memory
    checkCudaErrors(cudaFree(d_arr1));
    checkCudaErrors(cudaFree(d_arr2));
    checkCudaErrors(cudaFree(d_result));
}

}