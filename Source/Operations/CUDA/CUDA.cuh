// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <cuda_runtime.h>

namespace Flow
{
    __device__
    inline int MultiToFlatIndex_Device( int* index, int* shape, int shapeSize )
    {
        int flatIndex = 0;
        int stride = 1;
        for ( int i = shapeSize - 1; i >= 0; i-- )
        {
            flatIndex += index[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

    __device__
    inline void FlatToMultiIndex_Device( int index, int* shape, int shapeSize, int* multiIndex )
    {
        for ( int i = shapeSize - 1; i >= 0; i-- )
        {
            multiIndex[i] = index % shape[i];
            index /= shape[i];
        }
    }
}