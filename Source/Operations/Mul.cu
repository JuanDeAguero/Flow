// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArray.h"

__global__
void Mul_Kernel( float* arr1, float* arr2, float* result )
{
    int i = blockIdx.x;
    result[i] = arr1[i] * arr2[i];
}
    
NARRAY Flow::Mul( NARRAY arr1, NARRAY arr2 )
{
    vector<int> shape = BroadcastShapes( arr1->GetShape(), arr2->GetShape() );
    NARRAY arr1B = Broadcast( arr1, shape );
    NARRAY arr2B = Broadcast( arr2, shape );
    int n = SizeFromShape(arr1B->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    Mul_Kernel<<< n, 1 >>>( arr1B->GetData(), arr2B->GetData(), result_d );
    cudaDeviceSynchronize();
    return Create( arr1B->GetShape(), result_d, { arr1B, arr2B }, NArray::Operation::MUL );
}

NARRAY Flow::Mul( NARRAY arr, float literal )
{
    return Mul( arr, Create( { 1 }, { literal } ) );
}

__global__
void BackwardMul_Kernel( float* gradient, float* operand1, float* operandGradient1, float* operand2,
    float* operandGradient2 )
{
    int i = blockIdx.x;
    atomicAdd( &operandGradient1[i], operand2[i] * gradient[i] );
    atomicAdd( &operandGradient2[i], operand1[i] * gradient[i] );
}

void Flow::NArray::BackwardMul()
{
    int n = SizeFromShape(Gradient->GetShape());
    BackwardMul_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(),
        Operands[0]->GetGradient()->GetData(), Operands[1]->GetData(),
        Operands[1]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}