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

        enum Operation { NONE, ADD, SUB, MULT, DIV, MMULT, POW, EXP, TANH };

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

        static int SizeFromShape( vector<int> shape );

        NArray* Copy();

    private:

        vector<float> Data;

        vector<int> Shape;

        bool Constant;

        NArray* Gradient;

        vector<NArray*> Operands;
        
        Operation Op;

        void Backward();

        void BackwardAdd();

        void BackwardSub();

        void BackwardMult();

        void BackwardMMult();

        void BackwardPow();

        void BackwardExp();

        void BackwardTanh();

        vector<NArray*> TopologicalSort();

        void BuildTopo( NArray* current, unordered_set<NArray*>& visited, vector<NArray*>& topo );

        void BuildGraph( NArray* current, set<NArray*>& nodes, set< pair< NArray*, NArray* > >& edges );

    };

    NArray* ElementWise( NArray* arr1, NArray* arr2, NArray::Operation op );

    NArray* Add( NArray* arr1, NArray* arr2 );

    NArray* Sub( NArray* arr1, NArray* arr2 );

    NArray* Mult( NArray* arr1, NArray* arr2 );

    NArray* Mult( NArray* arr, float literal );

    NArray* MMult( NArray* arr1, NArray* arr2 );

    NArray* Div( NArray* arr1, NArray* arr2 );

    NArray* Pow( NArray* arr, float exponent );

    NArray* Exp( NArray* arr );

    NArray* Tanh( NArray* arr );

    NArray* Neg( NArray* arr );

    bool Less( NArray* arr1, NArray* arr2 );

    string OperationString( NArray::Operation op );

    NArray* Random( vector<int> shape );
}