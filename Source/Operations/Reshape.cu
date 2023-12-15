// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"
    
Flow::NArrayCore* Flow::Reshape( NArrayCore* arr, vector<int> shape )
{
    float* result_d;
    cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );
    cudaMemcpy( result_d, arr->GetData(), SizeFromShape(arr->GetShape()) * sizeof(float), cudaMemcpyDeviceToDevice );
    return new NArrayCore( shape, result_d, { arr }, NArrayCore::Operation::UNSQUEEZE );
}

__global__
void BackwardReshape_Kernel( float* gradient, float* operandGradient )
{
    int i = blockIdx.x;
    operandGradient[i] += gradient[i];
}

void Flow::NArrayCore::BackwardReshape()
{
    int n = SizeFromShape(Shape);
    BackwardReshape_Kernel<<< n, 1 >>>( Gradient->GetData(), Operands[0]->GetGradient()->GetData() );   
}