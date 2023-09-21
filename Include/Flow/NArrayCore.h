// Copyright (c) 2023 Juan M. G. de Agüero

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#pragma once

namespace Flow
{
    using namespace std;

    class NArrayCore
    {

    public:

        float Exponent;

        int TransposedFirstDim, TransposedSecondDim;

        NArrayCore( vector<int> shape, vector<float> data );

        enum Operation
        {
            NONE,
            ADD, MUL,
            MM,
            POW, EXP, TANH, RELU, LOG, RECIPROCAL,
            SUM, MAX,
            RESHAPE, TRANSPOSE, PERMUTE, EXPAND,
            BROADCAST
        };

        NArrayCore( vector<int> shape, vector<float> data, vector<NArrayCore*> operands, Operation op );

        float Get( vector<int> coordinates );

        vector<float> Get();

        vector<int> GetShape();

        vector<int> GetStride();

        NArrayCore* GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Backpropagate();

        NArrayCore* Copy();

    private:

        vector<float> Data;

        vector<int> Shape;

        vector<int> Stride;

        NArrayCore* Gradient;

        vector<NArrayCore*> Operands;
        
        Operation Op;

        NArrayCore( vector<int> shape, vector<float> data, bool isGradient );

        int GetIndex( vector<int> coordinates );

        void ComputeStride();

        void Backward();

        void BackwardAdd();

        void BackwardMul();

        void BackwardMM();

        void BackwardPow();

        void BackwardExp();

        void BackwardTanh();

        void BackwardReshape();

        void BackwardTranspose();

        void BackwardBroadcast();

        vector<NArrayCore*> TopologicalSort();

        void BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo );
    };

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr, float literal );

    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape );

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim );

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape );

    NArrayCore* Neg( NArrayCore* arr );

    bool Less( NArrayCore* arr1, NArrayCore* arr2 );

    int SizeFromShape( vector<int> shape );

    vector<int> GetShapeForBroadcast( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* RandomCore( vector<int> shape );

    void Print( NArrayCore* arr );
}