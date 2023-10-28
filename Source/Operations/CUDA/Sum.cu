// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Sum_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* result, int* resultShape, int resultShapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
        multiIndex[dim] = 0;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, resultShape, resultShapeSize );
        atomicAdd( &result[flatIndex], arr[i] );
    }

    __host__
    NArrayCore* Sum_CUDA( NArrayCore* arr, int dim )
    {
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = 1;
        vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
        int n = arr->Get().size();
        float* arr_d = HostToDeviceArr(arr);
        int* arrshape_d = HostToDeviceVec<int>(arr->GetShape());
        float* result_d = HostToDeviceVec<float>(resultData);
        int* resultshape_d = HostToDeviceVec<int>(resultShape);
        Sum_Kernel<<< n, 1 >>>( arr_d, arrshape_d, arr->GetShape().size(), dim, result_d, resultshape_d, resultShape.size() );
        cudaMemcpy( resultData.data(), result_d, resultData.size() * sizeof(float), cudaMemcpyDeviceToHost );
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::SUM );
        result->SumDim = dim;
        return result;
    }

    __global__
    void BackwardSum_Kernel( int dim, float* arr, int* shape, int shapeSize, float* operand, int* operandShape, int operandShapeSize, float* operandGradient, float* gradient )
    {
        int i = blockIdx.x;
        int j = blockIdx.y;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
        multiIndex[dim] = j;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
        atomicAdd( &operandGradient[flatIndex], gradient[i] );
    }

    __host__
    void Flow::NArrayCore::BackwardSum_CUDA()
    {
        int n = Data.size();
        int maxDimSize = Operands[0]->GetShape()[SumDim];
        float* arr_d = HostToDeviceVec<float>(Data);
        int* shape_d = HostToDeviceVec<int>(Shape);
        float* operand_d = HostToDeviceArr(Operands[0]);
        int* operandShape_d = HostToDeviceVec<int>(Operands[0]->GetShape());
        float* operandGradient_d = HostToDeviceArr(Operands[0]->Gradient);
        float* gradient_d = HostToDeviceArr(Gradient);
        dim3 gridDims( n, maxDimSize );
        BackwardSum_Kernel<<< gridDims, 1 >>>( SumDim, arr_d, shape_d, Shape.size(), operand_d, operandShape_d, Operands[0]->GetShape().size(), operandGradient_d, gradient_d );
        cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient_d, Operands[0]->Gradient->Data.size() * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr_d);
        cudaFree(shape_d);
        cudaFree(gradient_d);
        cudaFree(operand_d);
        cudaFree(operandShape_d);
        cudaFree(operandGradient_d);
    }
}