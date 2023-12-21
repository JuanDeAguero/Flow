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

        float* GetData();

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
    
    NArray Broadcast( NArray arr, vector<int> shape );
    
    NArray CrossEntropy( NArray arr1, NArray arr2 );
    
    NArray Div( NArray arr1, NArray arr2 );
    
    NArray Exp( NArray arr );
    
    NArray Gather( NArray arr, int dim, NArray index );
    
    NArray Index( NArray arr, int dim, NArray index );
    
    NArray Log( NArray arr );
    
    NArray Max( NArray arr, int dim );
    
    NArray Mean( NArray arr, int dim );
    
    NArray MM( NArray arr1, NArray arr2 );
    
    NArray Mul( NArray arr, float literal );
    
    NArray Mul( NArray arr1, NArray arr2 );
    
    NArray Neg( NArray arr );
    
    NArray Pow( NArray arr, float exponent );
    
    NArray ReLU( NArray arr );
    
    NArray Reshape( NArray arr, vector<int> shape );
    
    NArray Softmax( NArray arr, int dim );
    
    NArray Sub( NArray arr1, NArray arr2 );
    
    NArray Sum( NArray arr, int dim );
    
    NArray Tanh( NArray arr );
    
    NArray Transpose( NArray arr, int firstDim, int secondDim );
    
    NArray Unsqueeze( NArray arr, int dim );

    NArray Random( vector<int> shape );

    NArray Zeros( vector<int> shape );

    NArray Ones( vector<int> shape );

    NArray OneHot( vector<int> integers, int num );

    void Print( NArray arr );
}