// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Index_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int dim, Flow::NArrayDevice* index, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device(i, multiIndex, result);
    multiIndex[dim] = index->Data[multiIndex[dim]];
    int flatIndex = Flow::MultiToFlatIndex_Device(multiIndex, arr);
    int arrIndex = Flow::GetIndex_Device(flatIndex, arr);
    int resultIndex = Flow::GetIndex_Device(i, result);
    result->Data[resultIndex] = arr->Data[arrIndex];
}

NARRAY Flow::Index(NARRAY arr, int dim, NARRAY index) {
    vector<int> resultShape = arr->GetShape();
    resultShape[dim] = SizeFromShape(index->GetShape());
    NARRAY result = make_shared<NArray>(resultShape, vector<NARRAY>({ arr }), NArray::Operation::INDEX);
    result->IndexDim = dim;
    result->IndexIndex = index;
    int n = SizeFromShape(resultShape);
    Index_Kernel<<<BLOCKS(n), TPB>>>(arr->DeviceStruct, result->DeviceStruct, dim, index->DeviceStruct, n);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardIndex_Kernel(
    Flow::NArrayDevice* arr,
    Flow::NArrayDevice* grad,
    Flow::NArrayDevice* operand,
    Flow::NArrayDevice* operandGrad,
    int dim,
    Flow::NArrayDevice* index,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device(i, multiIndex, arr);
    multiIndex[dim] = index->Data[multiIndex[dim]];
    int flatIndex = Flow::MultiToFlatIndex_Device(multiIndex, operand);
    int operandGradIndex = Flow::GetIndex_Device(flatIndex, operandGrad);
    int gradIndex = Flow::GetIndex_Device(i, grad);
    Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], grad->Data[gradIndex]);
}

void Flow::NArray::BackwardIndex() {
    int n = SizeFromShape(Gradient->GetShape());
    BackwardIndex_Kernel<<<BLOCKS(n), TPB>>>(
        DeviceStruct,
        Gradient->DeviceStruct,
        Operands[0]->DeviceStruct,
        Operands[0]->Gradient->DeviceStruct,
        IndexDim,
        IndexIndex->DeviceStruct,
        n
    );
    CUDA_DeviceSynchronize();
}