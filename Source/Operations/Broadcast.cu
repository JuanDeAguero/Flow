// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Broadcast_Kernel( float* arr, int* arrShape, int arrShapeSize, int* shape, int shapeSize, float* result )
    {
        int i = blockIdx.x;
        int position[10];
        FlatToMultiIndex_Device( i, shape, shapeSize, position );
        int originalCoords[10];
        for ( int j = 0; j < arrShapeSize; j++ )
        {
            int coord = position[ shapeSize - arrShapeSize + j ];
            if ( arrShape[j] == 1 ) coord = 0;
            originalCoords[j] = coord;
        }
        int flatIndex = MultiToFlatIndex_Device( originalCoords, arrShape, arrShapeSize );
        result[i] = arr[flatIndex];
    }

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
        float* result_ptr = (float*)malloc( n * sizeof(float) );
        cudaMemcpy( result_ptr, result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        vector<float> resultData = vector<float>( result_ptr, result_ptr + n );
        return new NArrayCore( shape, resultData, { arr }, NArrayCore::Operation::BROADCAST );
    }
}