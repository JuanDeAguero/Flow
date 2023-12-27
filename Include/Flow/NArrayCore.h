// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace Flow
{
    using namespace std;

    class NArray;

    class NArrayCore
    {

    public:

        NArrayCore( vector<int> shape, const vector<float>& data );

        enum Operation
        {
            NONE,
            ADD,     BMM,     BROADCAST, CROSSENTROPY,
            DIV,     DOT,     EXP,       GATHER,
            INDEX,   LOG,     MATMUL,    MAX,
            MEAN,    MM,      MUL,       MV,
            NEG,     POW,     PROD,      RELU,
            RESHAPE, SOFTMAX, SQUEEZE,   SUB,
            SUM,     TANH,    TRANSPOSE, UNSQUEEZE
        };

        NArrayCore( vector<int> shape, float* deviceData, vector<NArrayCore*> operands, Operation op );

        ~NArrayCore();

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

        void Destroy();

    private:

        friend class NArray;

        float* Data;

        vector<int> Shape;

        NArrayCore* Gradient;

        vector<NArrayCore*> Operands;
        
        Operation Op;

        bool Saved;

        float Exponent;

        NArrayCore( vector<int> shape );  // contructor for gradients

        vector<NArrayCore*> TopologicalSort();

        void BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo );

        void Backward();

        void BackwardAdd();
        
        void BackwardBroadcast();
        
        void BackwardExp();
        
        void BackwardGather();
        
        void BackwardIndex();
        
        void BackwardLog();
        
        void BackwardMax();
        
        void BackwardMM();
        
        void BackwardMul();
        
        void BackwardPow();
        
        void BackwardReLU();
        
        void BackwardReshape();
        
        void BackwardSum();
        
        void BackwardTanh();
        
        void BackwardTranspose();
        
        void BackwardUnsqueeze();

        friend NArrayCore* Pow( NArrayCore* arr, float exponent );

    public:
        
        int GatherDim;

        NArrayCore* GatherIndex;
        
        int MaxDim;
        
        int SumDim;
        
        int TransposeFirstDim, TransposeSecondDim;

        int IndexDim;

        NArrayCore* Index;

    };

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape );

    NArrayCore* CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Log( NArrayCore* arr );

    NArrayCore* Matmul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Max( NArrayCore* arr, int dim );

    NArrayCore* Mean( NArrayCore* arr, int dim );

    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr, float literal );

    NArrayCore* Neg( NArrayCore* arr );
    
    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* ReLU( NArrayCore* arr );

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape );

    NArrayCore* Softmax( NArrayCore* arr, int dim );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Sum( NArrayCore* arr, int dim );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim );
    
    NArrayCore* Unsqueeze( NArrayCore* arr, int dim );

    NArrayCore* Random( vector<int> shape );

    NArrayCore* Zeros( vector<int> shape );

    NArrayCore* Ones( vector<int> shape );

    NArrayCore* OneHot( vector<int> integers, int num );

    void Print( NArrayCore* arr );

    void PrintShape( NArrayCore* arr );

    int SizeFromShape( vector<int> shape );

    int MultiToFlatIndex( vector<int> index, vector<int> shape );

    vector<int> FlatToMultiIndex( int index, vector<int> shape );

    vector<int> BroadcastShapes( vector<int> shape1, vector<int> shape2 );

    float GetCUDAFreeMemory();
}