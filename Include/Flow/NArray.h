// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#define NARRAY std::shared_ptr<Flow::NArrayCore>

namespace Flow
{
    using namespace std;

    namespace NArray
    {
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
    }

    class NArrayCore
    {

    public:

        NArrayCore( vector<int> shape, const vector<float>& data );

        NArrayCore( vector<int> shape, float* deviceData, vector<NARRAY> operands,
            NArray::Operation op );

        NArrayCore( vector<int> shape );  // Constructor for gradients.

        ~NArrayCore();

        float Get( vector<int> coordinates );

        vector<float> Get();

        float* GetData();

        vector<int> GetShape();
        
        int* GetShapeData();

        NARRAY GetGradient();
        
        void Set( vector<int> coordinates, float value );

        void Reset( float value );

        void Backpropagate();

        NARRAY Copy();

        void Copy( NARRAY arr );

    private:

        friend class NARRAY;

        float* Data;

        vector<int> Shape;

        NARRAY Gradient;

        vector<NARRAY> Operands;
        
        NArray::Operation Op;

        bool Saved;

        int GatherDim;

        NARRAY GatherIndex;

        int IndexDim;

        NARRAY Index;
        
        int MaxDim;

        float Exponent;
        
        int SumDim;
        
        int TransposeFirstDim, TransposeSecondDim;

        vector<NArrayCore*> TopologicalSort();

        void BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited,
            vector<NArrayCore*>& topo );

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

        friend NARRAY Gather( NARRAY arr, int dim, NARRAY index );

        friend NARRAY Index( NARRAY arr, int dim, NARRAY index );

        friend NARRAY Max( NARRAY arr, int dim );

        friend NARRAY Pow( NARRAY arr, float exponent );
        
        friend NARRAY Sum( NARRAY arr, int dim );

        friend NARRAY Transpose( NARRAY arr, int firstDim, int secondDim );

    };

    namespace NArray
    {
        NARRAY Create( vector<int> shape, const vector<float>& data );

        NARRAY Create( vector<int> shape, float* deviceData, vector<NARRAY> operands,
            NArray::Operation op );
    }

    NARRAY Add( NARRAY arr1, NARRAY arr2 );

    NARRAY Broadcast( NARRAY arr, vector<int> shape );

    NARRAY CrossEntropy( NARRAY arr1, NARRAY arr2 );

    NARRAY Div( NARRAY arr1, NARRAY arr2 );

    NARRAY Exp( NARRAY arr );

    NARRAY Gather( NARRAY arr, int dim, NARRAY index );

    NARRAY Index( NARRAY arr, int dim, NARRAY index );

    NARRAY Log( NARRAY arr );

    NARRAY Matmul( NARRAY arr1, NARRAY arr2 );

    NARRAY Max( NARRAY arr, int dim );

    NARRAY Mean( NARRAY arr, int dim );

    NARRAY MM( NARRAY arr1, NARRAY arr2 );

    NARRAY Mul( NARRAY arr1, NARRAY arr2 );

    NARRAY Mul( NARRAY arr, float literal );

    NARRAY Neg( NARRAY arr );
    
    NARRAY Pow( NARRAY arr, float exponent );

    NARRAY ReLU( NARRAY arr );

    NARRAY Reshape( NARRAY arr, vector<int> shape );

    NARRAY Softmax( NARRAY arr, int dim );

    NARRAY Sub( NARRAY arr1, NARRAY arr2 );

    NARRAY Sum( NARRAY arr, int dim );

    NARRAY Tanh( NARRAY arr );

    NARRAY Transpose( NARRAY arr, int firstDim, int secondDim );
    
    NARRAY Unsqueeze( NARRAY arr, int dim );

    NARRAY Random( vector<int> shape );

    NARRAY Zeros( vector<int> shape );

    NARRAY Ones( vector<int> shape );

    NARRAY OneHot( vector<int> integers, int num );

    void Print( NARRAY arr );

    void PrintShape( NARRAY arr );

    int SizeFromShape( vector<int> shape );

    int MultiToFlatIndex( vector<int> index, vector<int> shape );

    vector<int> FlatToMultiIndex( int index, vector<int> shape );

    vector<int> BroadcastShapes( vector<int> shape1, vector<int> shape2 );

    float GetCUDAFreeMemory();
}