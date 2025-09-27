// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include <cmath>

#include "../Utils/CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Pow_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* result, float exp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int arrIndex = Flow::GetIndex_Device(i, arr);
    int resultIndex = Flow::GetIndex_Device(i, result);
    result->Data[resultIndex] = pow(arr->Data[arrIndex], exp);
}

NARRAY Flow::Pow(NARRAY arr, float exp) {
    NARRAY result = make_shared<NArray>(arr->GetShape(), vector<NARRAY>({ arr }), NArray::Operation::POW);
    result->Exponent = exp;
    int n = SizeFromShape(arr->GetShape());
    Pow_Kernel<<<BLOCKS(n), TPB>>>(arr->GetDeviceStruct(), result->GetDeviceStruct(), exp, n);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardPow_Kernel(Flow::NArrayDevice* grad, Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, float exp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int gradIndex = Flow::GetIndex_Device(i, grad);
    int operandIndex = Flow::GetIndex_Device(i, operand);
    int operandGradIndex = Flow::GetIndex_Device(i, operandGrad);
    float gradient = grad->Data[gradIndex] * exp * pow(operand->Data[operandIndex], exp - 1);
    Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], gradient);
}

void Flow::NArray::BackwardPow() {
    int n = SizeFromShape(Shape);
    BackwardPow_Kernel<<<BLOCKS(n), TPB>>>(Gradient->DeviceStruct, Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, Exponent, n);
    CUDA_DeviceSynchronize();
}