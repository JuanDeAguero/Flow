// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "NArrayCore.h"

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

    __device__
    inline float AtomicMax_Device( float* address, float val )
    {
        int* addressInt = (int*) address;
        int old = *addressInt, assumed;
        do
        {
            assumed = old;
            old = atomicCAS( addressInt, assumed, __float_as_int( fmaxf( val, __int_as_float(assumed) ) ) );
        }
        while ( assumed != old );
        return __int_as_float(old);
    }

    template< class T >
    inline T* HostToDeviceVec( std::vector<T> vec )
    {
        T* ptr;
        cudaMalloc( (void**)&ptr, vec.size() * sizeof(T) );
        cudaMemcpy( ptr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice );
        return ptr;
    }

    inline float* HostToDeviceArr( NArrayCore* arr )
    {
        float* ptr;
        cudaMalloc( (void**)&ptr, arr->Get().size() * sizeof(float) );
        cudaMemcpy( ptr, arr->GetData(), arr->Get().size() * sizeof(float), cudaMemcpyHostToDevice );
        return ptr;
    }
}