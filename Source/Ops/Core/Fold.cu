// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "../Utils/CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Fold2d_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int batchSize, int channels, int kernelHeight, int kernelWidth, int outHeight, int outWidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= outHeight || j >= outWidth) return;
    for (int b = 0; b < batchSize; b++)
    for (int c = 0; c < channels; c++)
    for (int ki = 0; ki < kernelHeight; ki++)
    for (int kj = 0; kj < kernelWidth; kj++) {
        int arrMultiIndex[MAX_DIMS];
        arrMultiIndex[0] = b;
        arrMultiIndex[1] = (c * kernelHeight * kernelWidth) + (ki * kernelWidth) + kj;
        arrMultiIndex[2] = (i * outWidth) + j;
        int resultMultiIndex[MAX_DIMS];
        resultMultiIndex[0] = b;
        resultMultiIndex[1] = c;
        resultMultiIndex[2] = i + ki;
        resultMultiIndex[3] = j + kj;
        int arrIndex = Flow::MultiToFlatIndex_Device(arrMultiIndex, arr);
        int resultIndex = Flow::MultiToFlatIndex_Device(resultMultiIndex, result);
        Flow::AtomicAdd_Device(&result->Data[resultIndex], arr->Data[arrIndex]);
    }
}

NARRAY Flow::Fold2d(NARRAY arr, vector<int> outShape, vector<int> kernel) {
    int batchSize = outShape[0];
    int channels = outShape[1];
    int height = outShape[2];
    int width = outShape[3];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    vector<int> resultShape = { batchSize, channels, height, width };
    NARRAY result = make_shared<NArray>(resultShape, vector<NARRAY>({ arr }), NArray::Operation::FOLD2D);
    result->FoldOutShape2d = outShape;
    result->FoldKernel2d = kernel;
    dim3 threadsPerBlock(16, 16);
    int blocksX = (outWidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (outHeight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(blocksX, blocksY);
    Fold2d_Kernel<<<numBlocks, threadsPerBlock>>>(arr->DeviceStruct, result->DeviceStruct, batchSize, channels, kernelHeight, kernelWidth, outHeight, outWidth);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardFold2d_Kernel(
    Flow::NArrayDevice* arr, 
    Flow::NArrayDevice* grad, 
    Flow::NArrayDevice* operand, 
    Flow::NArrayDevice* operandGrad, 
    int batchSize, 
    int channels, 
    int kernelHeight, 
    int kernelWidth, 
    int outHeight, 
    int outWidth
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= outHeight || j >= outWidth) return;
    for (int b = 0; b < batchSize; b++)
    for (int c = 0; c < channels; c++)
    for (int ki = 0; ki < kernelHeight; ki++)
    for (int kj = 0; kj < kernelWidth; kj++) {
        int arrMultiIndex[MAX_DIMS];
        arrMultiIndex[0] = b;
        arrMultiIndex[1] = c;
        arrMultiIndex[2] = i + ki;
        arrMultiIndex[3] = j + kj;
        int operandMultiIndex[MAX_DIMS];
        operandMultiIndex[0] = b;
        operandMultiIndex[1] = (c * kernelHeight * kernelWidth) + (ki * kernelWidth) + kj;
        operandMultiIndex[2] = (i * outWidth) + j;
        int arrIndex = Flow::MultiToFlatIndex_Device(arrMultiIndex, arr);
        int operandIndex = Flow::MultiToFlatIndex_Device(operandMultiIndex, operand);
        int operandGradIndex = Flow::GetIndex_Device(operandIndex, operandGrad);
        int gradIndex = Flow::GetIndex_Device(arrIndex, grad);
        Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], grad->Data[gradIndex]);
    }
}

void Flow::NArray::BackwardFold2d() {
    int batchSize = FoldOutShape2d[0];
    int channels = FoldOutShape2d[1];
    int height = FoldOutShape2d[2];
    int width = FoldOutShape2d[3];
    int kernelHeight = FoldKernel2d[0];
    int kernelWidth = FoldKernel2d[1];
    int outHeight = height - kernelHeight + 1;
    int outWidth = width - kernelWidth + 1;
    dim3 threadsPerBlock(16, 16);
    int blocksX = (outWidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (outHeight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(blocksX, blocksY);
    BackwardFold2d_Kernel<<<numBlocks, threadsPerBlock>>>(
        DeviceStruct, 
        Gradient->DeviceStruct, 
        Operands[0]->DeviceStruct, 
        Operands[0]->Gradient->DeviceStruct, 
        batchSize, 
        channels, 
        kernelHeight, 
        kernelWidth, 
        outHeight, 
        outWidth
    );
    CUDA_DeviceSynchronize();
}