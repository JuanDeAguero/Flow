// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

__global__
void Broadcast_Kernel( float* arr, int* arrShape, int arrShapeSize, int* shape, int shapeSize, float* result )
{
    int i = blockIdx.x;
    int position[10];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, position );
    int originalCoords[10];
    for ( int j = 0; j < arrShapeSize; j++ )
    {
        int coord = position[ shapeSize - arrShapeSize + j ];
        if ( arrShape[j] == 1 ) coord = 0;
        originalCoords[j] = coord;
    }
    int flatIndex = Flow::MultiToFlatIndex_Device( originalCoords, arrShape, arrShapeSize );
    result[i] = arr[flatIndex];
}

namespace Flow
{
    __host__
    NArrayCore* Broadcast_CUDA( NArrayCore* arr, vector<int> shape )
    {
        int n = SizeFromShape(shape);
        float* arr_d;
        int* arrShape_d;
        int* shape_d;
        float* result_d;
        cudaMalloc( (void**)&arr_d, arr->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&shape_d, shape.size() * sizeof(int) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMemcpy( arr_d, arr->GetData(), arr->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( shape_d, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice );
        Broadcast_Kernel<<< n, 1 >>>( arr_d, arrShape_d, arr->GetShape().size(), shape_d, shape.size(), result_d );
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr_d);
        cudaFree(arrShape_d);
        cudaFree(shape_d);
        cudaFree(result_d);
        return new NArrayCore( shape, resultData, { arr }, NArrayCore::Operation::BROADCAST );
    }
}

__global__
void BackwardBroadcast_Kernel_A( int* shape, int shapeSize, int* operandShape, int operandShapeSize, float* newOperandGradient, float* gradient )
{
    int i = blockIdx.x;
    int position[10];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, position );
    int operandCoords[10];
    for ( int j = 0; j < operandShapeSize; j++ )
    {
        int coord = position[ shapeSize - operandShapeSize + j ];
        if ( operandShape[j] == 1 ) coord = 0;
        operandCoords[j] = coord;
    }
    int operandIndex = Flow::MultiToFlatIndex_Device( operandCoords, operandShape, operandShapeSize );
    atomicAdd( &newOperandGradient[operandIndex], gradient[i] );
}

__global__
void BackwardBroadcast_Kernel_B( float* operandGradient, float* newOperandGradient )
{
    int i = blockIdx.x;
    operandGradient[i] += newOperandGradient[i];
}

__host__
void Flow::NArrayCore::BackwardBroadcast_CUDA()
{
    vector<float> newOperandGradient( Operands[0]->Data.size(), 0.0f );
    int n = Gradient->Data.size();
    int* shape_d;
    int* operandShape_d;
    float* newOperandGradient_d;
    float* gradient_d;
    cudaMalloc( (void**)&shape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&newOperandGradient_d, newOperandGradient.size() * sizeof(float) );
    cudaMalloc( (void**)&gradient_d, Gradient->Get().size() * sizeof(float) );
    cudaMemcpy( shape_d, GetShapeData(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(), Operands[0]->Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( newOperandGradient_d, newOperandGradient.data(), newOperandGradient.size() * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( gradient_d, Gradient->GetData(), Gradient->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
    BackwardBroadcast_Kernel_A<<< n, 1 >>>( shape_d, Shape.size(), operandShape_d, Operands[0]->Shape.size(), newOperandGradient_d, gradient_d );
    n = Operands[0]->Gradient->Data.size();
    float* operandGradient_d;
    cudaMalloc( (void**)&operandGradient_d, Operands[0]->Gradient->Get().size() * sizeof(float) );
    cudaMemcpy( operandGradient_d, Operands[0]->Gradient->GetData(), Operands[0]->Gradient->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
    BackwardBroadcast_Kernel_B<<< n, 1 >>>( operandGradient_d, newOperandGradient_d );
    cudaMemcpy( Operands[0]->Gradient->GetData(), operandGradient_d, Operands[0]->Gradient->Get().size() * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(shape_d);
    cudaFree(operandShape_d);
    cudaFree(newOperandGradient_d);
    cudaFree(gradient_d);
    cudaFree(operandGradient_d);
}