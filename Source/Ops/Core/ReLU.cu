// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "../Utils/CUDA.cuh"
#include "Flow/NArray.h"

__global__
void ReLU_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int arrIndex = Flow::GetIndex_Device(i, arr);
    int resultIndex = Flow::GetIndex_Device(i, result);
    result->Data[resultIndex] = arr->Data[arrIndex];
    Flow::AtomicMax_Device(&result->Data[resultIndex], 0.0f);
}

NARRAY Flow::ReLU(NARRAY arr) {
    NARRAY result = make_shared<NArray>(arr->GetShape(), vector<NARRAY>({ arr }), NArray::Operation::RELU);
    int n = SizeFromShape(arr->GetShape());
    ReLU_Kernel<<<BLOCKS(n), TPB>>>(arr->GetDeviceStruct(), result->GetDeviceStruct(), n);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardReLU_Kernel(Flow::NArrayDevice* grad, Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int gradIndex = Flow::GetIndex_Device(i, grad);
    int operandIndex = Flow::GetIndex_Device(i, operand);
    int operandGradIndex = Flow::GetIndex_Device(i, operandGrad);
    float gradient = 0.0f;
    if (operand->Data[operandIndex] > 0.0f) gradient = grad->Data[gradIndex];
    Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], gradient);
}

void Flow::NArray::BackwardReLU() {
    int n = SizeFromShape(Shape);
    BackwardReLU_Kernel<<<BLOCKS(n), TPB>>>(Gradient->DeviceStruct, Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, n);
    CUDA_DeviceSynchronize();
}