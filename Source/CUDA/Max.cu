// Copyright (c) 2023 Juan M. G. de Agüero

#include <limits>

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Max_Kernel( float* arr, int* arrShape, int arrShapeSize, int dim, float* result, int* resultShape, int resultShapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
        multiIndex[dim] = 0;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, resultShape, resultShapeSize );
        Flow::AtomicMax_Device( &result[flatIndex], arr[i] );
    }

    __host__
    NArrayCore* Max_CUDA( NArrayCore* arr, int dim )
    {
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = 1;
        vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
        int n = arr->Get().size();
        float* arr_d = HostToDeviceArr(arr);
        int* arrshape_d = HostToDeviceVec<int>(arr->GetShape());
        float* result_d = HostToDeviceVec<float>(resultData);
        int* resultshape_d = HostToDeviceVec<int>(resultShape);
        Max_Kernel<<< n, 1 >>>( arr_d, arrshape_d, arr->GetShape().size(), dim, result_d, resultshape_d, resultShape.size() );
        cudaMemcpy( resultData.data(), result_d, resultData.size() * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(arr_d);
        cudaFree(arrshape_d);
        cudaFree(result_d);
        cudaFree(resultshape_d);
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::MAX );
        result->MaxDim = dim;
        return result;
    }

    __global__
    void BackwardMax_Kernel( float* gradient, float* operand, int* operandShape, int operandShapeSize, float* operandGradient, float* arr, int* shape, int shapeSize, int dim )
    {
        int i = blockIdx.x;
        int j = blockIdx.y;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, shape, shapeSize, multiIndex );
        multiIndex[dim] = j;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, operandShape, operandShapeSize );
        if ( operand[flatIndex] == arr[i] )
            atomicAdd( &operandGradient[flatIndex], gradient[i] );
    }

    __host__
    void Flow::NArrayCore::BackwardMax_CUDA()
    {
        int n = Data.size();
        int maxDimSize = Operands[0]->GetShape()[MaxDim];
        float* gradient_d = HostToDeviceArr(Gradient);
        float* operand_d = HostToDeviceArr(Operands[0]);
        int* operandShape_d = HostToDeviceVec<int>(Operands[0]->GetShape());
        float* operandGradient_d = HostToDeviceArr(Operands[0]->Gradient);
        float* arr_d = HostToDeviceVec<float>(Data);
        int* shape_d = HostToDeviceVec<int>(Shape);
        dim3 gridDims( n, maxDimSize );
        BackwardMax_Kernel<<< gridDims, 1 >>>( gradient_d, operand_d, operandShape_d, Operands[0]->GetShape().size(), operandGradient_d, arr_d, shape_d, Shape.size(), MaxDim );
        cudaMemcpy( Operands[0]->Gradient->Data.data(), operandGradient_d, Operands[0]->Gradient->Data.size() * sizeof(float), cudaMemcpyDeviceToHost );
        cudaFree(gradient_d);
        cudaFree(operand_d);
        cudaFree(operandShape_d);
        cudaFree(operandGradient_d);
        cudaFree(arr_d);
        cudaFree(shape_d);
    }
}