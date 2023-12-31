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
    cudaDeviceSynchronize();
    cudaFree(arrShape_d);
    cudaFree(shape_d);
    return new NArrayCore( shape, result_d, { arr }, NArrayCore::Operation::BROADCAST );
}

__global__
void BackwardBroadcast_Kernel( float* gradient, int* shape, int shapeSize, int* operandShape, int operandShapeSize, float* operandGradient )
{
    int i = blockIdx.x;
    int position[MAX_DIMS];
    Flow::FlatToMultiIndex_Device( i, shape, shapeSize, position );
    int operandCoords[MAX_DIMS];
    for ( int j = 0; j < operandShapeSize; j++ )
    {
        int coord = position[ shapeSize - operandShapeSize + j ];
        operandCoords[j] = ( operandShape[j] == 1 ) ? 0 : coord;
    }
    int operandIndex = Flow::MultiToFlatIndex_Device( operandCoords, operandShape, operandShapeSize );
    atomicAdd( &operandGradient[operandIndex], gradient[i] );
}

__host__
void Flow::NArrayCore::BackwardBroadcast()
{
    int n = SizeFromShape(Gradient->GetShape());
    int* shape_d;
    int* operandShape_d;
    cudaMalloc( (void**)&shape_d, Shape.size() * sizeof(int) );
    cudaMalloc( (void**)&operandShape_d, Operands[0]->Shape.size() * sizeof(int) );
    cudaMemcpy( shape_d, GetShapeData(), Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( operandShape_d, Operands[0]->GetShapeData(), Operands[0]->Shape.size() * sizeof(int), cudaMemcpyHostToDevice );
    BackwardBroadcast_Kernel<<< n, 1 >>>( Gradient->GetData(), shape_d, Shape.size(), operandShape_d, Operands[0]->Shape.size(), Operands[0]->Gradient->GetData() );
    cudaDeviceSynchronize();
    cudaFree(shape_d);
    cudaFree(operandShape_d);
}