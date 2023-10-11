// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArrayCore.h"

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

    NArray Mul( NArray arr1, NArray arr2 );

    NArray Mul( NArray arr, float literal );

    NArray MM( NArray arr1, NArray arr2 );

    NArray Pow( NArray arr, float exponent );

    NArray Exp( NArray arr );

    NArray Tanh( NArray arr );

    NArray ReLU( NArray arr );

    NArray Log( NArray arr );

    NArray Sum( NArray arr, int dim );

    NArray Max( NArray arr, int dim );

    NArray Transpose( NArray arr, int firstDim, int secondDim );

    NArray Broadcast( NArray arr, vector<int> shape );

    NArray Gather( NArray arr, int dim, NArray index );

    NArray Unsqueeze( NArray arr, int dim );

    NArray Index( NArray arr, int dim, NArray index );

    NArray Neg( NArray arr );

    NArray Sub( NArray arr1, NArray arr2 );

    NArray Div( NArray arr1, NArray arr2 );

    NArray Mean( NArray arr );

    NArray Softmax( NArray arr );

    NArray CrossEntropy( NArray arr1, NArray arr2 );

    NArray Random( vector<int> shape );

    NArray Ones( vector<int> shape );

    void Print( NArray arr );
}