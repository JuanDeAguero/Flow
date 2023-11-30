// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Index_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, int* indices, float* result, int* resultShape, int resultShapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, resultShape, resultShapeSize, multiIndex );
        multiIndex[dim] = indices[multiIndex[dim]];
        int flatIndex = MultiToFlatIndex_Device( multiIndex, arrShape, arrShapeSize );
        result[i] = arr[flatIndex];
    }

    __host__
    NArrayCore* Index_CUDA( NArrayCore* arr, int dim, NArrayCore* index )
    {
        vector<int> indices(index->Get().size());
        for ( int i = 0; i < index->Get().size(); i++ )
            indices[i] = static_cast<int>(index->Get()[i]);
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = indices.size();
        int resultSize = 1;
        for ( int size : resultShape ) resultSize *= size;
        int n = resultSize;
        float* arr_d;
        int* arrShape_d;
        int* indices_d;
        float* result_d;
        int* resultShape_d;
        cudaMalloc( (void**)&arr_d, arr->Get().size() * sizeof(float) );
        cudaMalloc( (void**)&arrShape_d, arr->GetShape().size() * sizeof(int) );
        cudaMalloc( (void**)&indices_d, indices.size() * sizeof(int) );
        cudaMalloc( (void**)&result_d, n * sizeof(float) );
        cudaMalloc( (void**)&resultShape_d, resultShape.size() * sizeof(int) );
        cudaMemcpy( arr_d, arr->GetData(), arr->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( arrShape_d, arr->GetShapeData(), arr->GetShape().size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( indices_d, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( result_d, arr->GetData(), n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( resultShape_d, resultShape.data(), resultShape.size() * sizeof(int), cudaMemcpyHostToDevice );
        Index_Kernel<<< n, 1 >>>( arr_d, arrShape_d, arr->GetShape().size(), dim, indices_d, result_d, resultShape_d, resultShape.size() );
        vector<float> resultData(n);
        cudaMemcpy( resultData.data(), result_d, n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr_d);
        cudaFree(arrShape_d);
        cudaFree(indices_d);
        cudaFree(result_d);
        cudaFree(resultShape_d);
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr, index }, NArrayCore::Operation::INDEX );
        result->IndexDim = dim;
        result->Index = index;
        return result;
    }

    __global__
    void BackwardIndex_Kernel( float* gradient, int* operandShape, int operandShapeSize, float* operandGradient, int dim, int* indices, int* shape, int shapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
        multiIndex[dim] = indices[multiIndex[dim]];
        int flatIndex = MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
        operandGradient[flatIndex] += gradient[i];
    }

    __host__
    void NArrayCore::BackwardIndex_CUDA()
    {
        vector<int> indices(Index->Get().size());
        for ( int i = 0; i < Index->Get().size(); i++ )
            indices[i] = static_cast<int>(Index->Get()[i]);
        int n = Gradient->Data.size();
        float* gradient_d = HostToDeviceArr(Gradient);
        int* operandShape_d = HostToDeviceVec<int>(Operands[0]->GetShape());
        float* operandGradient_d = HostToDeviceArr(Operands[0]->Gradient);
        int* indices_d = HostToDeviceVec<int>(indices);
        int* shape_d = HostToDeviceVec<int>(Shape);
        BackwardIndex_Kernel<<< n, 1 >>>( gradient_d, operandShape_d, Operands[0]->GetShape().size(), operandGradient_d, IndexDim, indices_d, shape_d, Shape.size() );
        cudaMemcpy( Operands[0]->Gradient->GetData(), operandGradient_d, Operands[0]->Gradient->Get().size() * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(gradient_d);
        cudaFree(operandShape_d);
        cudaFree(operandGradient_d);
        cudaFree(indices_d);
        cudaFree(shape_d);
    }
}