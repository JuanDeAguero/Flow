// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

#include <chrono>
#include "Flow/Print.h"

__global__
void Add_Kernel( float* arr1, float* arr2, float* result )
{
    int i = blockIdx.x;
    result[i] = arr1[i] + arr2[i];
}
    
Flow::NArrayCore* Flow::Add( NArrayCore* arr1, NArrayCore* arr2 )
{
    vector<int> shape = BroadcastShapes( arr1->GetShape(), arr2->GetShape() );
    NArrayCore* arr1B = Broadcast( arr1, shape );
    NArrayCore* arr2B = Broadcast( arr2, shape );
    int n = SizeFromShape(arr1B->GetShape());
    float* result_d;
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    Add_Kernel<<< n, 1 >>>( arr1B->GetData(), arr2B->GetData(), result_d );
    return new NArrayCore( arr1B->GetShape(), result_d, { arr1B, arr2B }, NArrayCore::Operation::ADD );
}

__global__
void BackwardAdd_Kernel( float* gradient, float* operandGradient1, float* operandGradient2 )
{
    int i = blockIdx.x;
    operandGradient1[i] += gradient[i];
    operandGradient2[i] += gradient[i];
}

void Flow::NArrayCore::BackwardAdd()
{
    int n = SizeFromShape(Gradient->GetShape());
    BackwardAdd_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetGradient()->GetData(), Operands[1]->GetGradient()->GetData() );
}