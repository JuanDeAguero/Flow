// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "../Utils/CUDA.cuh"
#include "Flow/NArray.h"

#define TILE_SIZE 32

__global__
void BMM_Kernel(Flow::NArrayDevice* arr1, Flow::NArrayDevice* arr2, Flow::NArrayDevice* result, int arr1Rows, int arr1Cols, int arr2Cols, int batchSize) {
    __shared__ float arr1_s[TILE_SIZE][TILE_SIZE];
    __shared__ float arr2_s[TILE_SIZE][TILE_SIZE];
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + threadY;
    int col = blockIdx.x * TILE_SIZE + threadX;
    int batchIndex = blockIdx.z;
    int arr1Offset = batchIndex * arr1Rows * arr1Cols;
    int arr2Offset = batchIndex * arr1Cols * arr2Cols;
    int resultOffset = batchIndex * arr1Rows * arr2Cols;
    float sum = 0.0f;
    for (int i = 0; i < ceil(arr1Cols / float(TILE_SIZE)); i++) {
        if (row < arr1Rows && (i * TILE_SIZE + threadX) < arr1Cols) {
            int tileXIndex = i * TILE_SIZE + threadX;
            int rowIndex = row * arr1Cols;
            int arr1Index = Flow::GetIndex_Device(arr1Offset + rowIndex + tileXIndex, arr1);
            arr1_s[threadY][threadX] = arr1->Data[arr1Index];
        }
        else arr1_s[threadY][threadX] = 0.0f;
        if ((i * TILE_SIZE + threadY) < arr1Cols && col < arr2Cols) {
            int tileYIndex = i * TILE_SIZE + threadY;
            int arr2Index = Flow::GetIndex_Device(arr2Offset + tileYIndex * arr2Cols + col, arr2);
            arr2_s[threadY][threadX] = arr2->Data[arr2Index];
        }
        else arr2_s[threadY][threadX] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) sum += arr1_s[threadY][k] * arr2_s[k][threadX];
        __syncthreads();
    }
    if (row < arr1Rows && col < arr2Cols) {
        int resultIndex = Flow::GetIndex_Device(resultOffset + row * arr2Cols + col, result);
        result->Data[resultIndex] = sum;
    }
}

NARRAY Flow::BMM(NARRAY arr1, NARRAY arr2) {
    int arr1Rows = arr1->GetShape()[1];
    int arr1Cols = arr1->GetShape()[2];
    int arr2Cols = arr2->GetShape()[2];
    int batchSize = arr1->GetShape()[0];
    vector<int> resultShape = { batchSize, arr1Rows, arr2Cols };
    NARRAY result = make_shared<NArray>(resultShape, vector<NARRAY>({ arr1, arr2 }), NArray::Operation::BMM);
    dim3 dimGrid(ceil(arr2Cols / float(TILE_SIZE)), ceil(arr1Rows / float(TILE_SIZE)), batchSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    BMM_Kernel<<<dimGrid, dimBlock>>>(arr1->GetDeviceStruct(), arr2->GetDeviceStruct(), result->GetDeviceStruct(), arr1Rows, arr1Cols, arr2Cols, batchSize);
    CUDA_DeviceSynchronize();
    return result;
}

void Flow::NArray::BackwardBMM() {
    Operands[0]->Gradient = Flow::BMM(Gradient->Copy(), Transpose(Operands[1]->Copy(), 1, 2));
    Operands[1]->Gradient = Flow::BMM(Transpose(Operands[0]->Copy(), 1, 2), Gradient->Copy());
}