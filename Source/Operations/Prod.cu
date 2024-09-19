// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Prod_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* result, int dim, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = 0;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, result );
    int resultIndex = Flow::GetIndex_Device( flatIndex, result );
    int arrIndex = Flow::GetIndex_Device( i, arr );
    Flow::AtomicMul_Device( &result->Data[resultIndex], arr->Data[arrIndex] );
}

NARRAY Flow::Prod( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->Shape;
    resultShape[dim] = 1;
    NARRAY result = make_shared<NArray>( resultShape, vector<NARRAY>({ arr }),
        NArray::Operation::PROD );
    result->ProdDim = dim;
    result->Reset(1.0f);
    int n = SizeFromShape(arr->Shape);
    Prod_Kernel<<< BLOCKS(n), TPB >>>( arr->DeviceStruct, result->DeviceStruct, dim, n );
    CUDA_DeviceSynchronize();
    return result;
}

__global__
void BackwardProd_Kernel( Flow::NArrayDevice* arr, Flow::NArrayDevice* grad,
    Flow::NArrayDevice* operand, Flow::NArrayDevice* operandGrad, int dim, int n1, int n2 )
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if ( i >= n1 || j >= n2 ) return;
    int multiIndex[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, multiIndex, arr );
    multiIndex[dim] = j;
    int flatIndex = Flow::MultiToFlatIndex_Device( multiIndex, operand );
    float prodWithoutCurrent = 1.0f;
    for ( int k = 0; k < operand->Shape[dim]; k++ )
    {
        if ( k == j ) continue;
        multiIndex[dim] = k;
        int operandIndex = Flow::MultiToFlatIndex_Device( multiIndex, operand );
        prodWithoutCurrent *= operand->Data[operandIndex];
    }
    int operandGradIndex = Flow::GetIndex_Device( flatIndex, operandGrad );
    int gradIndex = Flow::GetIndex_Device( i, grad );
    Flow::AtomicAdd_Device( &operandGrad->Data[operandGradIndex], grad->Data[gradIndex] *
        prodWithoutCurrent );
}

void Flow::NArray::BackwardProd()
{
    int n1 = SizeFromShape(Shape);
    int n2 = Operands[0]->Shape[ProdDim];
    BackwardProd_Kernel<<< dim3( n1, n2 ), 1 >>>( DeviceStruct, Gradient->DeviceStruct,
        Operands[0]->DeviceStruct, Operands[0]->Gradient->DeviceStruct, ProdDim, n1, n2 );
    CUDA_DeviceSynchronize();
}