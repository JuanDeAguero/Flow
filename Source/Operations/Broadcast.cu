// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

std::vector<int> Flow::BroadcastShapes( vector<int> shape1, vector<int> shape2 )
{
    int maxDims = max( shape1.size(), shape2.size() );
    while ( shape1.size() < maxDims ) shape1.insert( shape1.begin(), 1 );
    while ( shape2.size() < maxDims ) shape2.insert( shape2.begin(), 1 );
    std::vector<int> shape(maxDims);
    for ( int i = 0; i < maxDims; i++ )
    {
        if ( shape1[i] == shape2[i] ) shape[i] = shape1[i];
        else if ( shape1[i] == 1 ) shape[i] = shape2[i];
        else if ( shape2[i] == 1 ) shape[i] = shape1[i];
    }
    return shape;
}

__global__
void Broadcast_Kernel( float* arr, int* arrShape, int arrShapeSize, int* shape, int shapeSize, float* result )
{
    int i = blockIdx.x;
    int position[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, position );
    int originalCoords[MAX_DIMS];
    for ( int j = 0; j < arrShapeSize; j++ )
    {
        int coord = position[ shapeSize - arrShapeSize + j ];
        if ( arrShape[j] == 1 ) coord = 0;
        originalCoords[j] = coord;
    }
    int flatIndex = Flow::MultiToFlatIndex_Device( originalCoords, arrShape, arrShapeSize );
    result[i] = arr[flatIndex];
}

Flow::NArrayCore* Flow::Broadcast( NArrayCore* arr, vector<int> shape )
{
    int n = SizeFromShape(shape);
    int* arrShape_d;
    int* shape_d;
    float* result_d;
    cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
    cudaMalloc( (void**)&shape_d, shape.size() * sizeof(int) );
    cudaMalloc( (void**)&result_d, n * sizeof(float) );
    cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( shape_d, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    Broadcast_Kernel<<< n, 1 >>>( arr->GetData(), arrShape_d, arr->GetShape().size(), shape_d, shape.size(), result_d );
    return new NArrayCore( shape, result_d, { arr }, NArrayCore::Operation::BROADCAST );
}

__global__
void BackwardBroadcast_Kernel_A( int* shape, int shapeSize, float* gradient, int* operandShape, int operandShapeSize, float* gradientIncrement )
{
    int i = blockIdx.x;
    int position[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, position );
    int operandCoords[MAX_DIMS];
    for ( int j = 0; j < operandShapeSize; j++ )
    {
        int coord = position[ shapeSize - operandShapeSize + j ];
        if ( operandShape[j] == 1 ) coord = 0;
        operandCoords[j] = coord;
    }
    int operandIndex = Flow::MultiToFlatIndex_Device( operandCoords, operandShape, operandShapeSize );
    atomicAdd( &gradientIncrement[operandIndex], gradient[i] );
}

__global__
void BackwardBroadcast_Kernel_B( float* operandGradient, float* gradientIncrement )
{
    int i = blockIdx.x;
    operandGradient[i] += gradientIncrement[i];
}

__host__
void Flow::NArrayCore::BackwardBroadcast()
{
    int n = SizeFromShape(Gradient->GetShape());
    int* shape_d;
    int* operandShape_d;
    float* gradientIncrement_d;
    cudaMalloc( (void**)&shape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&gradientIncrement_d, SizeFromShape(Operands[0]->GetShape()) * sizeof(float) );
    cudaMemcpy( shape_d, GetShapeData(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(), Operands[0]->Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemset( gradientIncrement_d, SizeFromShape(Operands[0]->GetShape()) * sizeof(float), 0.0f );
    BackwardBroadcast_Kernel_A<<< n, 1 >>>( shape_d, Shape.size(), Gradient->GetData(), operandShape_d, Operands[0]->Shape.size(), gradientIncrement_d );
    n = SizeFromShape(Operands[0]->GetGradient()->GetShape());
    BackwardBroadcast_Kernel_B<<< n, 1 >>>( Operands[0]->Gradient->GetData(), gradientIncrement_d );
}