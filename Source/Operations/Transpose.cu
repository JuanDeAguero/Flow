// Copyright (c) 2023 Juan M. G. de Agüero

#include "CUDA.cuh"
#include "Flow/NArrayCore.h"

namespace Flow
{
    __global__
    void Transpose_Kernel( float* arr, int* arrShape, int arrShapeSize, int firstDim, int secondDim, float* result, int* resultShape, int resultShapeSize )
    {
        int i = blockIdx.x;
        int multiIndex[10];
        FlatToMultiIndex_Device( i, arrShape, arrShapeSize, multiIndex );
        int temp = multiIndex[firstDim];
        multiIndex[firstDim] = multiIndex[secondDim];
        multiIndex[secondDim] = temp;
        int flatIndex = MultiToFlatIndex_Device( multiIndex, resultShape, resultShapeSize );
        result[flatIndex] = arr[i];
    }

    __host__
    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {
        return nullptr;

        /*auto start = chrono::high_resolution_clock::now();

        vector<int> resultShape = arr->GetShape();
        int temp = resultShape[firstDim];
        resultShape[firstDim] = resultShape[secondDim];
        resultShape[secondDim] = temp;
        //vector<float> resultData( SizeFromShape(arr->GetShape()), 0 );
        int n = SizeFromShape(arr->GetShape());
        //float* arr_d = HostToDeviceArr(arr);
        int* arrshape_d = HostToDeviceVec<int>(arr->GetShape());

        float* result_d;
        cudaMalloc( (void**)&result_d, SizeFromShape(arr->GetShape()) * sizeof(float) );

        int* resultshape_d = HostToDeviceVec<int>(resultShape);
        Transpose_Kernel<<< n, 1 >>>( arr->DeviceData, arrshape_d, arr->GetShape().size(), firstDim, secondDim, result_d, resultshape_d, resultShape.size() );
        //cudaMemcpy( resultData.data(), result_d, resultData.size() * sizeof(float), cudaMemcpyDeviceToHost );

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>( end - start );
        //Print( to_string(duration.count()) + " Transpose" );

        NArrayCore* result = new NArrayCore( resultShape, {}, { arr }, NArrayCore::Operation::TRANSPOSE );
        result->DeviceData = result_d;
        return result;*/
    }
}

void Flow::NArrayCore::BackwardTranspose()
{

}