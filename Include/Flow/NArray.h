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

        NArray( const NArray& arr ) = delete;

        NArray& operator=( const NArray& arr ) = delete;

        NArray( NArray&& arr ) = delete;

        NArray& operator=( NArray&& arr ) = delete;

        ~NArray();

        NArrayCore* GetCore();

        float Get( vector<int> coordinates );

        vector<float> Get();

        float* GetData();

        vector<int> GetShape();

        NArrayCore* GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void ResetGradient( float value );

        void Backpropagate();

        NArrayCore* Copy();

        NArrayCore* CopyGradient();

        void Assign( NArrayCore* arr );

    private:

        NArrayCore* Array;

    };

    NArrayCore* Create( vector<int> shape, const vector<float>& data );

    NArrayCore* Add( NArray& arr1, NArray& arr2 );

    NArrayCore* Add( NArrayCore* arr1, NArray& arr2 );
    
    NArrayCore* Broadcast( NArray& arr, vector<int> shape );
    
    NArrayCore* CrossEntropy( NArray& arr1, NArray& arr2 );
    
    NArrayCore* Div( NArray& arr1, NArray& arr2 );
    
    NArrayCore* Exp( NArray& arr );
    
    NArrayCore* Gather( NArray& arr, int dim, NArray& index );
    
    NArrayCore* Index( NArray& arr, int dim, NArray& index );
    
    NArrayCore* Log( NArray& arr );
    
    NArrayCore* Max( NArray& arr, int dim );
    
    NArrayCore* Mean( NArray& arr, int dim );
    
    NArrayCore* MM( NArray& arr1, NArray& arr2 );
    
    NArrayCore* Mul( NArray& arr, float literal );
    
    NArrayCore* Mul( NArray& arr1, NArray& arr2 );
    
    NArrayCore* Neg( NArray& arr );
    
    NArrayCore* Pow( NArray& arr, float exponent );
    
    NArrayCore* ReLU( NArray& arr );
    
    NArrayCore* Reshape( NArray& arr, vector<int> shape );
    
    NArrayCore* Softmax( NArray& arr, int dim );
    
    NArrayCore* Sub( NArray& arr1, NArray& arr2 );
    
    NArrayCore* Sum( NArray& arr, int dim );
    
    NArrayCore* Tanh( NArray& arr );
    
    NArrayCore* Transpose( NArray& arr, int firstDim, int secondDim );
    
    NArrayCore* Unsqueeze( NArray& arr, int dim );

    void Print( NArray& arr );
}