// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Add_Kernel( float* arr1, float* arr2, float* result )
{
    int i = blockIdx.x;
    result[i] = arr1[i] + arr2[i];
}
    
NARRAY Flow::Add( NARRAY arr1, NARRAY arr2 )
{
    vector<int> shape = BroadcastShapes( arr1->GetShape(), arr2->GetShape() );
    NARRAY arr1B = Broadcast( arr1, shape );
    NARRAY arr2B = Broadcast( arr2, shape );
    int n = SizeFromShape(arr1B->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    Add_Kernel<<< n, 1 >>>( arr1B->GetData(), arr2B->GetData(), result_d );
    return Create( arr1B->GetShape(), result_d, { arr1B, arr2B }, NArray::Operation::ADD );
}

__global__
void BackwardAdd_Kernel( float* gradient, float* operandGradient1, float* operandGradient2 )
{
    int i = blockIdx.x;
    atomicAdd( &operandGradient1[i], gradient[i] );
    atomicAdd( &operandGradient2[i], gradient[i] );
}

void Flow::NArray::BackwardAdd()
{
    int n = SizeFromShape(Gradient->GetShape());
    BackwardAdd_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetGradient()->GetData(),
        Operands[1]->GetGradient()->GetData() );
}