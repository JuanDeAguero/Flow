// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArrayCore.h"

#pragma once

namespace Flow
{
    using namespace std;

    class NArray
    {

    public:

        NArray();

        NArray( NArrayCore* arr );

        bool IsValid();

        NArrayCore* GetCore();

        float Get( vector<int> coordinates );

        vector<float> Get();

        vector<int> GetShape();

        NArray GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Backpropagate();

        NArray Copy();

    private:

        NArrayCore* Array;

    };

    NArray Create( vector<int> shape, vector<float> data );

    NArray Add( NArray arr1, NArray arr2 );

    NArray Sub( NArray arr1, NArray arr2 );

    NArray Mul( NArray arr1, NArray arr2 );

    NArray Mul( NArray arr, float literal );

    NArray MM( NArray arr1, NArray arr2 );

    NArray Div( NArray arr1, NArray arr2 );

    NArray Pow( NArray arr, float exponent );

    NArray Exp( NArray arr );

    NArray Tanh( NArray arr );

    NArray Transpose( NArray arr, int firstDim, int secondDim );

    NArray Neg( NArray arr );

    bool Less( NArray arr1, NArray arr2 );

    NArray Random( vector<int> shape );

    void Print( NArray arr );
}