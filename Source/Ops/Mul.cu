// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Mul_Kernel(Flow::NArrayDevice* arr1, Flow::NArrayDevice* arr2, Flow::NArrayDevice* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int arr1Index = Flow::GetIndex_Device(i, arr1);
    int arr2Index = Flow::GetIndex_Device(i, arr2);
    int resultIndex = Flow::GetIndex_Device(i, result);
    result->Data[resultIndex] = arr1->Data[arr1Index] * arr2->Data[arr2Index];
}
    
NARRAY Flow::Mul(NARRAY arr1, NARRAY arr2) {
    vector<int> resultShape = BroadcastShapes(arr1->GetShape(), arr2->GetShape());
    NARRAY arr1B = Broadcast(arr1, resultShape);
    NARRAY arr2B = Broadcast(arr2, resultShape);
    NARRAY result = make_shared<NArray>(resultShape, vector<NARRAY>({ arr1B, arr2B }), NArray::Operation::MUL);
    int n = SizeFromShape(resultShape);
    Mul_Kernel<<<BLOCKS(n), TPB>>>(arr1B->GetDeviceStruct(), arr2B->GetDeviceStruct(), result->GetDeviceStruct(), n);
    CUDA_DeviceSynchronize();
    return result;
}

NARRAY Flow::Mul(NARRAY arr, float literal) {
    return Mul(arr, Create({ 1 }, { literal }));
}

__global__
void BackwardMul_Kernel(Flow::NArrayDevice* grad, Flow::NArrayDevice* operand1, Flow::NArrayDevice* operandGrad1, Flow::NArrayDevice* operand2, Flow::NArrayDevice* operandGrad2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int gradIndex = Flow::GetIndex_Device(i, grad);
    int operandIndex1 = Flow::GetIndex_Device(i, operand1);
    int operandGradIndex1 = Flow::GetIndex_Device(i, operandGrad1);
    int operandIndex2 = Flow::GetIndex_Device(i, operand2);
    int operandGradIndex2 = Flow::GetIndex_Device(i, operandGrad2);
    Flow::AtomicAdd_Device(&operandGrad1->Data[operandGradIndex1], operand2->Data[operandIndex2] * grad->Data[gradIndex]);
    Flow::AtomicAdd_Device(&operandGrad2->Data[operandGradIndex2], operand1->Data[operandIndex1] * grad->Data[gradIndex]);
}

void Flow::NArray::BackwardMul() {
    int n = SizeFromShape(Gradient->Shape);
    BackwardMul_Kernel<<<BLOCKS(n), TPB>>>(Gradient->DeviceStruct, Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, Operands[1]->DeviceStruct, Operands[1]->Gradient->DeviceStruct, n);
    CUDA_DeviceSynchronize();
}