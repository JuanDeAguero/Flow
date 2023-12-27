// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Mul_Kernel( float* arr1, float* arr2, float* result )
{
    int i = blockIdx.x;
    result[i] = arr1[i] * arr2[i];
}
    
Flow::NArrayCore* Flow::Mul( NArrayCore* arr1, NArrayCore* arr2 )
{
    vector<int> shape = BroadcastShapes( arr1->GetShape(), arr2->GetShape() );
    NArrayCore* arr1B = Broadcast( arr1, shape );
    NArrayCore* arr2B = Broadcast( arr2, shape );
    int n = SizeFromShape(arr1B->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    Mul_Kernel<<< n, 1 >>>( arr1B->GetData(), arr2B->GetData(), result_d );
    cudaDeviceSynchronize();
    return new NArrayCore( arr1B->GetShape(), result_d, { arr1B, arr2B }, NArrayCore::Operation::MUL );
}

Flow::NArrayCore* Flow::Mul( NArrayCore* arr, float literal )
{
    NArrayCore* arrLiteral = new NArrayCore( { 1 }, { literal } );
    return Mul( arr, arrLiteral );
}

__global__
void BackwardMul_Kernel( float* gradient, float* operand1, float* operandGradient1, float* operand2, float* operandGradient2 )
{
    int i = blockIdx.x;
    atomicAdd( &operandGradient1[i], operand2[i] * gradient[i] );
    atomicAdd( &operandGradient2[i], operand1[i] * gradient[i] );
}

void Flow::NArrayCore::BackwardMul()
{
    int n = SizeFromShape(Gradient->GetShape());
    BackwardMul_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetData(), Operands[0]->GetGradient()->GetData(), Operands[1]->GetData(), Operands[1]->GetGradient()->GetData() );
    cudaDeviceSynchronize();
}