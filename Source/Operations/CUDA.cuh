// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "NArray.h"

#define MAX_DIMS 100
#define TPB 256
#define BLOCKS(n) ( n + TPB - 1 ) / TPB

namespace Flow
{
    using namespace std;

    __global__
    void Reset_Kernel( float* arr, float value, int n );

    __device__
    inline void FlatToMultiIndex_Device( int flatIndex, int* multiIndex, NArrayDevice* arr )
    {
        int product = 1;
        for ( int i = arr->ShapeSize - 1; i >= 0; i-- ) product *= arr->Shape[i];
        for ( int i = 0; i < arr->ShapeSize; i++ )
        {
            product /= arr->Shape[i];
            multiIndex[i] = ( flatIndex / product ) % arr->Shape[i];
        }
    }

    __device__
    inline int MultiToFlatIndex_Device( int* multiIndex, NArrayDevice* arr )
    {
        int flatIndex = arr->Offset;
        for ( int i = arr->ShapeSize - 1; i >= 0; i-- ) flatIndex += multiIndex[i] * arr->Stride[i];
        return flatIndex;
    }

        __device__
    inline int GetIndex_Device( int linearIndex, NArrayDevice* arr )
    {
        int multiIndex[MAX_DIMS];
        FlatToMultiIndex_Device( linearIndex, multiIndex, arr );
        return MultiToFlatIndex_Device( multiIndex, arr );
    }

    __device__
    inline void AtomicAdd_Device( float* address, float value )
    {
        atomicAdd( address, value );
    }

    __device__
    inline float AtomicMul_Device( float* address, int value )
    {
        int* addressInt = (int*)address;
        int old = *addressInt, assumed;
        do
        {
            assumed = old;
            old = atomicCAS( addressInt, assumed,
                __float_as_int( value * __int_as_float(assumed) ) );
        }
        while ( assumed != old );
        return __int_as_float(old);
    }

    __device__
    inline float AtomicMax_Device( float* address, float value )
    {
        int* addressInt = (int*)address;
        int old = *addressInt, assumed;
        do
        {
            assumed = old;
            old = atomicCAS( addressInt, assumed,
                __float_as_int( fmaxf(value, __int_as_float(assumed) ) ) );
        }
        while ( assumed != old );
        return __int_as_float(old);
    }

    pair< vector<int>, float* > BMMRaw( pair< vector<int>, float* > arr1,
        pair< vector<int>, float* > arr2 );

    pair< vector<int>, float* > BMMRaw( NARRAY arr1, NARRAY arr2 );

    pair< vector<int>, float* > BMMRaw( pair< vector<int>, float* > arr1, NARRAY arr2 );

    pair< vector<int>, float* > BMMRaw( NARRAY arr1, pair< vector<int>, float* > arr2 );

    pair< vector<int>, float* > TransposeRaw( NARRAY arr, int firstDim, int secondDim );
}