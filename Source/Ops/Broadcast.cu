// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include <stdexcept>

#include "CUDA.cuh"
#include "Flow/NArray.h"

using namespace std;

vector<int> Flow::BroadcastShapes(vector<int> shape1, vector<int> shape2) {
    int maxDims = max(shape1.size(), shape2.size());
    while (shape1.size() < maxDims) shape1.insert(shape1.begin(), 1);
    while (shape2.size() < maxDims) shape2.insert(shape2.begin(), 1);
    vector<int> shape(maxDims);
    for (int i = 0; i < maxDims; i++) {
        if (shape1[i] == shape2[i]) shape[i] = shape1[i];
        else if (shape1[i] == 1) shape[i] = shape2[i];
        else if (shape2[i] == 1) shape[i] = shape1[i];
        else throw runtime_error("Incompatible shapes for broadcast!");
    }
    return shape;
}

NARRAY Flow::Broadcast(NARRAY arr, vector<int> shape) {
    if (arr->GetShape() == shape) return arr;
    vector<int> arrShape = arr->GetShape();
    vector<int> resultShape(max(arrShape.size(), shape.size()), 1);
    vector<int> resultStride(resultShape.size(), 0);
    for (int i = resultShape.size() - 1, j = arrShape.size() - 1, k = shape.size() - 1; i >= 0; i--, j--, k--) {
        int originalDim = j >= 0 ? arrShape[j] : 1;
        int targetDim = k >= 0 ? shape[k] : 1;
        resultShape[i] = max(originalDim, targetDim);
        resultStride[i] = (originalDim == 1) ? 0 : (j >= 0 ? arr->GetStride()[j] : 0);
    }
    return make_shared<NArray>(resultShape, resultStride, arr->GetOffset(), FindMetaParent(arr), vector<NARRAY>({ arr }), NArray::Operation::BROADCAST);
}

__global__
void BackwardBroadcast_Kernel(Flow::NArrayDevice* arr, Flow::NArrayDevice* grad, Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device(i, multiIndex, arr);
    int operandCoords[MAX_DIMS];
    for (int j = 0; j < operand->ShapeSize; j++) {
        int coord = multiIndex[ arr->ShapeSize - operand->ShapeSize + j ];
        if (operand->Shape[j] == 1) operandCoords[j] = 0;
        else operandCoords[j] = coord;
    }
    int flatIndex = Flow::MultiToFlatIndex_Device(operandCoords, operand);
    int operandGradIndex = Flow::GetIndex_Device(flatIndex, operandGrad);
    int gradIndex = Flow::GetIndex_Device(i, grad);
    Flow::AtomicAdd_Device(&operandGrad->Data[operandGradIndex], grad->Data[gradIndex]);
}

void Flow::NArray::BackwardBroadcast() {
    int n = SizeFromShape(Gradient->Shape);
    BackwardBroadcast_Kernel<<<BLOCKS(n), TPB>>>(DeviceStruct, Gradient->DeviceStruct, Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, n);
    CUDA_DeviceSynchronize();
}