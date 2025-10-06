// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Gather_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int resultSize, int dim, Flow::NArrayDevice* index, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device(i, multiIndex, index);
    int indexIndex = Flow::GetIndex_Device(i, index);
    if (index->Data[indexIndex] < 0 || index->Data[indexIndex] >= arr->Shape[dim]) return;
    multiIndex[dim] = (int)index->Data[indexIndex];
    int flatIndex = Flow::MultiToFlatIndex_Device(multiIndex, arr);
    if (flatIndex < 0 || flatIndex >= resultSize) return;
    int arrIndex = Flow::GetIndex_Device(flatIndex, arr);
    int resultIndex = Flow::GetIndex_Device(i, result);
    result->Data[resultIndex] = arr->Data[arrIndex];
}

NARRAY Flow::Gather(NARRAY arr, int dim, NARRAY index) {
    NARRAY result = make_shared<NArray>(index->Shape, vector<NARRAY>({ arr }), NArray::Operation::GATHER);
    result->GatherDim = dim;
    result->GatherIndex = index;
    int n = SizeFromShape(result->Shape);
    Gather_Kernel<<<BLOCKS(n), TPB>>>(arr->DeviceStruct, result->DeviceStruct, SizeFromShape(arr->Shape), dim, index->DeviceStruct, n);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardGather_Kernel(Flow::NArrayDevice* grad, Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int operandGradSize, int dim, Flow::NArrayDevice* index, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device(i, multiIndex, index);
    int indexIndex = Flow::GetIndex_Device(i, index);
    int indexElement = (int)index->Data[indexIndex];
    if (indexElement < 0 || indexElement >= operand->Shape[dim]) return;
    multiIndex[dim] = indexElement;
    int flatIndex = Flow::MultiToFlatIndex_Device(multiIndex, operand);
    if (flatIndex < 0 || flatIndex >= operandGradSize) return;
    int operandGradIndex = Flow::GetIndex_Device(flatIndex, operandGrad);
    int gradIndex = Flow::GetIndex_Device(i, grad);
    Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], grad->Data[gradIndex]);
}

void Flow::NArray::BackwardGather() {
    int n = SizeFromShape(GatherIndex->GetShape());
    BackwardGather_Kernel<<<BLOCKS(n), TPB>>>(
        Gradient->DeviceStruct, 
        Operands[0]->DeviceStruct, 
        Operands[0]->Gradient->DeviceStruct, 
        SizeFromShape(Operands[0]->Gradient->Shape), 
        GatherDim, 
        GatherIndex->DeviceStruct, 
        n
    );
    CUDA_DeviceSynchronize();
}