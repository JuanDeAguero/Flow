// Copyright (c) 2023 Juan M. G. de Agüero

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#pragma once

namespace Flow
{
    using namespace std;

    class NArray
    {

    public:

        NArray( vector<int> shape, vector<float> data );

        NArray( vector<int> shape, vector<float> data, bool constant );

        enum Operation { NONE, ADD, SUB, MULT, MMULT, POW, TANH, EXP };

        NArray( vector<int> shape, vector<float> data, vector<NArray*> operands, Operation op );

        float Get( vector<int> coordinates );

        vector<float> Get();

        vector<int> GetShape();

        int GetIndex( vector<int> coordinates );

        NArray* GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Reshape( vector<int> shape );

        void Backpropagate();

        string Trace();

        // TODO: private
        vector<float> Data;

    private:

        vector<int> Shape;

        bool Constant;

        NArray* Gradient;

        vector<NArray*> Operands;
        
        Operation Op;

        void SetStrides();

        void Backward();

        void BackwardAdd();

        void BackwardSub();

        void BackwardMult();

        void BackwardMMult();

        void BackwardPow();

        void BackwardTanh();

        void BackwardExp();

        void BackwardSum();

        vector<NArray*> TopologicalSort();

        void BuildTopo( NArray* current, unordered_set<NArray*>& visited, vector<NArray*>& topo );

        void BuildGraph( NArray* current, set<NArray*>& nodes, set< pair< NArray*, NArray* > >& edges );

        int SizeFromShape( vector<int> shape );

    };

    NArray* Add( NArray* arr1, NArray* arr2 );

    NArray* Sub( NArray* arr1, NArray* arr2 );

    NArray* Neg( NArray* arr );

    NArray* Mult( NArray* arr1, NArray* arr2 );

    NArray* Mult( NArray* arr, float literal );

    NArray* MMult( NArray* arr1, NArray* arr2 );

    NArray* Div( NArray* arr1, NArray* arr2 );

    NArray* Pow( NArray* arr, float exponent );

    bool Less( NArray* arr1, NArray* arr2 );

    NArray* Tanh( NArray* arr );

    NArray* Exp( NArray* arr );

    string OperationString( NArray::Operation op );

    NArray* Random( vector<int> shape );
}