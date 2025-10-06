// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Unfold2d_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int batchSize, int channels, int kernelHeight, int kernelWidth, int strideHeight, int strideWidth, int outHeight, int outWidth) {
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
        arrMultiIndex[2] = (i * strideHeight) + ki;
        arrMultiIndex[3] = (j * strideWidth) + kj;
        int resultMultiIndex[MAX_DIMS];
        resultMultiIndex[0] = b;
        resultMultiIndex[1] = (c * kernelHeight * kernelWidth) + (ki * kernelWidth) + kj;
        resultMultiIndex[2] = (i * outWidth) + j;
        int arrIndex = Flow::MultiToFlatIndex_Device(arrMultiIndex, arr);
        int resultIndex = Flow::MultiToFlatIndex_Device(resultMultiIndex, result);
        result->Data[resultIndex] = arr->Data[arrIndex];
    }
}

NARRAY Flow::Unfold2d(NARRAY arr, vector<int> kernel, vector<int> stride) {
    int batchSize = arr->Shape[0];
    int channels = arr->Shape[1];
    int height = arr->Shape[2];
    int width = arr->Shape[3];
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int strideHeight = stride[0];
    int strideWidth = stride[1];
    int outHeight = ((height - kernelHeight) / strideHeight) + 1;
    int outWidth = ((width - kernelWidth) / strideWidth) + 1;
    vector<int> resultShape = { batchSize, channels * kernelHeight * kernelWidth, outHeight * outWidth };
    NARRAY result = make_shared<NArray>(resultShape, vector<NARRAY>({ arr }), NArray::Operation::UNFOLD2D);
    result->UnfoldKernel2d = kernel;
    result->UnfoldStride2d = stride;
    dim3 threadsPerBlock(16, 16);
    int blocksX = (outWidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (outHeight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(blocksX, blocksY);
    Unfold2d_Kernel<<<numBlocks, threadsPerBlock>>>(arr->DeviceStruct, result->DeviceStruct, batchSize, channels, kernelHeight, kernelWidth, strideHeight, strideWidth, outHeight, outWidth);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardUnfold2d_Kernel(
    Flow::NArrayDevice* arr, 
    Flow::NArrayDevice* grad, 
    Flow::NArrayDevice* operand, 
    Flow::NArrayDevice* operandGrad, 
    int batchSize, 
    int channels, 
    int kernelHeight, 
    int kernelWidth, 
    int strideHeight, 
    int strideWidth, 
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
        arrMultiIndex[1] = (c * kernelHeight * kernelWidth) + (ki * kernelWidth) + kj;
        arrMultiIndex[2] = (i * outWidth) + j;
        int operandMultiIndex[MAX_DIMS];
        operandMultiIndex[0] = b;
        operandMultiIndex[1] = c;
        operandMultiIndex[2] = (i * strideHeight) + ki;
        operandMultiIndex[3] = (j * strideWidth) + kj;
        int arrIndex = Flow::MultiToFlatIndex_Device(arrMultiIndex, arr);
        int operandIndex = Flow::MultiToFlatIndex_Device(operandMultiIndex, operand);
        int operandGradIndex = Flow::GetIndex_Device(operandIndex, operandGrad);
        int gradIndex = Flow::GetIndex_Device(arrIndex, grad);
        Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], grad->Data[gradIndex]);
    }
}

void Flow::NArray::BackwardUnfold2d() {
    int batchSize = Operands[0]->Shape[0];
    int channels = Operands[0]->Shape[1];
    int height = Operands[0]->Shape[2];
    int width = Operands[0]->Shape[3];
    int kernelHeight = UnfoldKernel2d[0];
    int kernelWidth = UnfoldKernel2d[1];
    int strideHeight = UnfoldStride2d[0];
    int strideWidth = UnfoldStride2d[1];
    int outHeight = ((height - kernelHeight) / strideHeight) + 1;
    int outWidth = ((width - kernelWidth) / strideWidth) + 1;
    dim3 threadsPerBlock(16, 16);
    int blocksX = (outWidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (outHeight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(blocksX, blocksY);
    BackwardUnfold2d_Kernel<<<numBlocks, threadsPerBlock>>>(
        DeviceStruct, 
        Gradient->DeviceStruct, 
        Operands[0]->DeviceStruct, 
        Operands[0]->Gradient->DeviceStruct, 
        batchSize, 
        channels, 
        kernelHeight, 
        kernelWidth, 
        strideHeight, 
        strideWidth, 
        outHeight, 
        outWidth
    );
    CUDA_DeviceSynchronize();
}