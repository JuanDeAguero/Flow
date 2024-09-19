// Copyright (c) 2023 Juan M. G. de Agüero

#include <limits>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Max_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int dim, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = 0;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, result );
    int resultIndex = Flow::GetIndex_Device( flatIndex, result );
    int arrIndex = Flow::GetIndex_Device( i, arr );
    Flow::AtomicMax_Device( &result->Data[resultIndex], arr->Data[arrIndex] );
}

NARRAY Flow::Max( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->GetShape();
    resultShape[dim] = 1;
    NARRAY result = make_shared<NArray>( resultShape, vector<NARRAY>({ arr }),
        NArray::Operation::MAX );
    result->MaxDim = dim;
    result->Reset( -1.0f * numeric_limits<float>::max() );
    int n = SizeFromShape(arr->GetShape());
    Max_Kernel<<< BLOCKS(n), TPB >>>( arr->DeviceStruct, result->DeviceStruct, dim, n);
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardMax_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* grad,
    Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int dim, int n1, int n2 )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if ( i >= n1 || j >= n2 ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = j;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operand );
    int operandIndex = Flow::GetIndex_Device( flatIndex, operand );
    int operandGradIndex = Flow::GetIndex_Device( flatIndex, operandGrad );
    int arrIndex = Flow::GetIndex_Device( i, arr );
    int gradIndex = Flow::GetIndex_Device( i, grad );
    if ( operand->Data[operandIndex] == arr->Data[arrIndex] )
        Flow::AtomicAdd_Device( &operandGrad->Data[operandGradIndex], grad->Data[gradIndex] );
}

void Flow::NArray::BackwardMax()
{
    int n1 = SizeFromShape(Shape);
    int n2 = Operands[0]->Shape[MaxDim];
    BackwardMax_Kernel<<< dim3( n1, n2 ), 1 >>>( DeviceStruct, Gradient->DeviceStruct,
        Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, MaxDim, n1, n2 );
    CUDA_DeviceSynchronize();
}