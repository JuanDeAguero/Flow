// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "NArrayCore.h"

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

        float Get( vector<int> coordinates );

        vector<float> Get();

        NArrayCore* GetCore();

        vector<int> GetShape();

        int GetIndex( vector<int> coordinates );

        NArray GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Reshape( vector<int> shape );

        void Backpropagate();

        NArray Copy();

    private:

        NArrayCore* Array;

    };

    NArray Create( vector<int> shape, vector<float> data );

    NArray Add( NArray arr1, NArray arr2 );

    NArray Sub( NArray arr1, NArray arr2 );

    NArray Mult( NArray arr1, NArray arr2 );

    NArray Mult( NArray arr, float literal );

    NArray MMult( NArray arr1, NArray arr2 );

    NArray Div( NArray arr1, NArray arr2 );

    NArray Pow( NArray arr, float exponent );

    NArray Exp( NArray arr );

    NArray Tanh( NArray arr );

    NArray Neg( NArray arr );

    bool Less( NArray arr1, NArray arr2 );

    NArray Random( vector<int> shape );

    void Log( NArray arr );
}