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

        NArrayCore( vector<int> shape, vector<float> data );

        enum Operation { NONE, ADD, SUB, MULT, DIV, MMULT, POW, EXP, TANH };

        NArrayCore( vector<int> shape, vector<float> data, vector<NArrayCore*> operands, Operation op );

        float Get( vector<int> coordinates );

        vector<float> Get();

        vector<int> GetShape();

        int GetIndex( vector<int> coordinates );

        NArrayCore* GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Reshape( vector<int> shape );

        void Backpropagate();

        NArrayCore* Copy();

        static int SizeFromShape( vector<int> shape );

    private:

        vector<float> Data;

        vector<int> Shape;

        NArrayCore* Gradient;

        vector<NArrayCore*> Operands;
        
        Operation Op;

        NArrayCore( vector<int> shape, vector<float> data, bool isGradient );

        void Backward();

        void BackwardAdd();

        void BackwardSub();

        void BackwardMult();

        void BackwardMMult();

        void BackwardPow();

        void BackwardExp();

        void BackwardTanh();

        vector<NArrayCore*> TopologicalSort();

        void BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo );

    };

    NArrayCore* ElementWise( NArrayCore* arr1, NArrayCore* arr2, NArrayCore::Operation op );

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mult( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mult( NArrayCore* arr, float literal );

    NArrayCore* MMult( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* Neg( NArrayCore* arr );

    bool Less( NArrayCore* arr1, NArrayCore* arr2 );

    string OperationString( NArrayCore::Operation op );

    NArrayCore* RandomCore( vector<int> shape );

    void Log( NArrayCore* arr );
}