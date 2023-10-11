// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

static bool UseCUDA = true;

namespace Flow
{
    using namespace std;

    class NArrayCore
    {

    public:

        NArrayCore( vector<int> shape, vector<float> data );

        enum Operation
        {
            NONE,
            ADD, MUL,
            MM,
            POW, EXP,
            TANH, RELU, LOG,
            SUM, MAX,
            RESHAPE, TRANSPOSE, BROADCAST,
            GATHER, SQUEEZE, UNSQUEEZE,
            INDEX
        };

        NArrayCore( vector<int> shape, vector<float> data, vector<NArrayCore*> operands, Operation op );

        float Get( vector<int> coordinates );

        vector<float> Get();

        float* GetData();

        vector<int> GetShape();
        
        int* GetShapeData();

        NArrayCore* GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Backpropagate();

        NArrayCore* Copy();

    private:

        vector<float> Data;

        vector<int> Shape;

        NArrayCore* Gradient;

        vector<NArrayCore*> Operands;
        
        Operation Op;

        NArrayCore( vector<int> shape, vector<float> data, bool isGradient );

        vector<NArrayCore*> TopologicalSort();

        void BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo );

        void Backward();

        void BackwardAdd();

        void BackwardAdd_CUDA();

        void BackwardMul();

        void BackwardMM();

        void BackwardPow();

        void BackwardExp();

        void BackwardExp_CUDA();

        void BackwardTanh();

        void BackwardReLU();

        void BackwardLog();

        void BackwardSum();

        void BackwardMax();

        void BackwardReshape();

        void BackwardTranspose();

        void BackwardBroadcast();

        void BackwardBroadcast_CUDA();

        void BackwardGather();

        void BackwardUnsqueeze();

        void BackwardIndex();

    public:

        float Exponent;

        int SumDim;
        
        int MaxDim;

        int TransposeFirstDim, TransposeSecondDim;

        int GatherDim;

        NArrayCore* GatherIndex;

        int UnsqueezeDim;

        int IndexDim;

        NArrayCore* Index;
    };

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr, float literal );

    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* ReLU( NArrayCore* arr );

    NArrayCore* Log( NArrayCore* arr );

    NArrayCore* Sum( NArrayCore* arr, int dim );

    NArrayCore* Max( NArrayCore* arr, int dim );

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape );

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim );

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape );

    NArrayCore* Broadcast_CUDA( NArrayCore* arr, vector<int> shape );

    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Unsqueeze( NArrayCore* arr, int dim );

    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Neg( NArrayCore* arr );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mean( NArrayCore* arr );

    NArrayCore* Softmax( NArrayCore* arr );

    NArrayCore* CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 );

    int SizeFromShape( vector<int> shape );

    int MultiToFlatIndex( vector<int> index, vector<int> shape );

    vector<int> FlatToMultiIndex( int index, vector<int> shape );

    vector<int> GetShapeForBroadcast( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* RandomCore( vector<int> shape );

    NArrayCore* OnesCore( vector<int> shape );

    void Print( NArrayCore* arr );
}