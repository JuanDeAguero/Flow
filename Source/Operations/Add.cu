// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

using namespace std;

__global__
void Add_Kernel( Flow::NArrayDevice* arr1, Flow::NArrayDevice* arr2, Flow::NArrayDevice* result,
    int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int arrIndex1 = Flow::GetIndex_Device( i, arr1 );
    int arrIndex2 = Flow::GetIndex_Device( i, arr2 );
    int resultIndex = Flow::GetIndex_Device( i, result );
    result->Data[resultIndex] = arr1->Data[arrIndex1] + arr2->Data[arrIndex2];
}

NARRAY Flow::Add( NARRAY arr1, NARRAY arr2 )
{
    vector<int> resultShape = BroadcastShapes( arr1->GetShape(), arr2->GetShape() );
    NARRAY arr1B = Broadcast( arr1, resultShape );
    NARRAY arr2B = Broadcast( arr2, resultShape );
    NARRAY result = make_shared<NArray>( resultShape, vector<NARRAY>({ arr1B, arr2B }),
        NArray::Operation::ADD );
    int n = SizeFromShape(resultShape);
    Add_Kernel<<< BLOCKS(n), TPB >>>( arr1B->GetDeviceStruct(), arr2B->GetDeviceStruct(),
        result->GetDeviceStruct(), n );
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardAdd_Kernel( Flow::NArrayDevice* grad, Flow::NArrayDevice* operandGrad1,
    Flow::NArrayDevice* operandGrad2, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int gradIndex = Flow::GetIndex_Device( i, grad );
    int operandGradIndex1 = Flow::GetIndex_Device( i, operandGrad1 );
    int operandGradIndex2 = Flow::GetIndex_Device( i, operandGrad2 );
    Flow::AtomicAdd_Device( &operandGrad1->Data[operandGradIndex1], grad->Data[gradIndex] );
    Flow::AtomicAdd_Device( &operandGrad2->Data[operandGradIndex2], grad->Data[gradIndex] );
}

void Flow::NArray::BackwardAdd()
{
    int n = SizeFromShape(Gradient->Shape);
    BackwardAdd_Kernel<<< BLOCKS(n), TPB >>>( Gradient->DeviceStruct,
        Operands[0]->Gradient->DeviceStruct, Operands[1]->Gradient->DeviceStruct, n );
    CUDA_DeviceSynchronize();
}