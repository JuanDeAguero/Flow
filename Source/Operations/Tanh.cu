// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Tanh_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int arrIndex = Flow::GetIndex_Device( i, arr );
    int resultIndex = Flow::GetIndex_Device( i, result );
    result->Data[resultIndex] = tanh(arr->Data[arrIndex]);
}

NARRAY Flow::Tanh( NARRAY arr )
{
    NARRAY result = make_shared<NArray>( arr->GetShape(), vector<NARRAY>({ arr }),
        NArray::Operation::TANH );
    int n = SizeFromShape(arr->GetShape());
    Tanh_Kernel<<< BLOCKS(n), TPB >>>( arr->GetDeviceStruct(), result->GetDeviceStruct(), n );
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardTanh_Kernel( Flow::NArrayDevice* grad, Flow::NArrayDevice* operand,
    Flow::NArrayDevice* operandGrad, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int gradIndex = Flow::GetIndex_Device( i, grad );
    int operandIndex = Flow::GetIndex_Device( i, operand );
    int operandGradIndex = Flow::GetIndex_Device( i, operandGrad );
    float value = tanh(operand->Data[operandIndex]);
    float gradient = grad->Data[gradIndex] * ( 1 - value * value );
    Flow::AtomicAdd_Device( &operandGrad->Data[operandGradIndex], gradient );
}

void Flow::NArray::BackwardTanh()
{
    int n = SizeFromShape(Shape);
    BackwardTanh_Kernel<<< BLOCKS(n), TPB >>>( Gradient->DeviceStruct, Operands[0]->DeviceStruct,
        Operands[0]->Gradient->DeviceStruct, n );
    CUDA_DeviceSynchronize();
}