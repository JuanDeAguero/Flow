// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include <algorithm>
#include <limits>

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Sum_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int dim, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = 0;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, result );
    int arrIndex = Flow::GetIndex_Device( i, arr );
    Flow::AtomicAdd_Device( &result->Data[flatIndex], arr->Data[arrIndex] );
}

NARRAY Flow::Sum( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->Shape;
    resultShape[dim] = 1;
    NARRAY result = make_shared<NArray>( resultShape, vector<NARRAY>({ arr }),
        NArray::Operation::SUM );
    result->SumDim = dim;
    int n = SizeFromShape(arr->Shape);
    Sum_Kernel<<< BLOCKS(n), TPB >>>( arr->DeviceStruct, result->DeviceStruct, dim, n );
    CUDA_DeviceSynchronize();
    return result;
}

NARRAY Flow::Sum( NARRAY arr, vector<int> dims )
{
    sort( dims.begin(), dims.end(), greater<int>() );
    for ( int dim : dims ) arr = Sum( arr, dim ) ;
    return arr;
}

__global__
void BackwardSum_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* grad,
    Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int dim, int n1, int n2 )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if ( i >= n1 || j >= n2 ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = j;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operand );
    int operandGradIndex = Flow::GetIndex_Device( flatIndex, operandGrad );
    int gradIndex = Flow::GetIndex_Device( i, grad );
    Flow::AtomicAdd_Device( &operandGrad->Data[operandGradIndex], grad->Data[gradIndex] );
}

void Flow::NArray::BackwardSum()
{
    int n1 = SizeFromShape(Shape);
    int n2 = Operands[0]->Shape[SumDim];
    BackwardSum_Kernel<<< dim3( n1, n2 ), 1 >>>( DeviceStruct, Gradient->DeviceStruct,
        Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, SumDim, n1, n2 );
    CUDA_DeviceSynchronize();
}