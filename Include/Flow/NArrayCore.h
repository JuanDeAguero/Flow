// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace Flow
{
    using namespace std;

    static bool UseCUDA = true;

    class NArrayCore
    {

    public:

        NArrayCore( vector<int> shape, vector<float> data );

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

        void BackwardAdd_CUDA();

        void BackwardAdd();
        
        void BackwardBroadcast_CUDA();
        
        void BackwardBroadcast();
        
        void BackwardExp_CUDA();
        
        void BackwardExp();
        
        void BackwardGather_CUDA();
        
        void BackwardGather();
        
        void BackwardIndex_CUDA();
        
        void BackwardIndex();
        
        void BackwardLog_CUDA();
        
        void BackwardLog();
        
        void BackwardMax_CUDA();
        
        void BackwardMax();
        
        void BackwardMM_CUDA();
        
        void BackwardMM();
        
        void BackwardMul_CUDA();
        
        void BackwardMul();
        
        void BackwardPow_CUDA();
        
        void BackwardPow();
        
        void BackwardReLU_CUDA();
        
        void BackwardReLU();
        
        void BackwardReshape();
        
        void BackwardSum_CUDA();
        
        void BackwardSum();
        
        void BackwardTanh_CUDA();
        
        void BackwardTanh();
        
        void BackwardTranspose();
        
        void BackwardUnsqueeze();

    public:

        float Exponent;
        
        int GatherDim;

        NArrayCore* GatherIndex;
        
        int MaxDim;
        
        int SumDim;
        
        int TransposeFirstDim, TransposeSecondDim;

        int IndexDim;

        NArrayCore* Index;
    };

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Add_CUDA( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Broadcast_CUDA( NArrayCore* arr, vector<int> shape );

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape );

    NArrayCore* CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Exp_CUDA( NArrayCore* arr );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Gather_CUDA( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Index_CUDA( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Log_CUDA( NArrayCore* arr );

    NArrayCore* Log( NArrayCore* arr );

    NArrayCore* Matmul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Max_CUDA( NArrayCore* arr, int dim );

    NArrayCore* Max( NArrayCore* arr, int dim );

    NArrayCore* Mean( NArrayCore* arr, int dim );

    NArrayCore* MM_CUDA( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mul( NArrayCore* arr, float literal );

    NArrayCore* Mul_CUDA( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Neg( NArrayCore* arr );

    NArrayCore* Pow_CUDA( NArrayCore* arr, float exponent );
    
    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* ReLU_CUDA( NArrayCore* arr );

    NArrayCore* ReLU( NArrayCore* arr );

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape );

    NArrayCore* Softmax( NArrayCore* arr, int dim );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Sum_CUDA( NArrayCore* arr, int dim );

    NArrayCore* Sum( NArrayCore* arr, int dim );

    NArrayCore* Tanh_CUDA( NArrayCore* arr );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim );

    NArrayCore* Transpose_CUDA( NArrayCore* arr, int firstDim, int secondDim );
    
    NArrayCore* Unsqueeze( NArrayCore* arr, int dim );

    NArrayCore* RandomCore( vector<int> shape );

    NArrayCore* ZerosCore( vector<int> shape );

    NArrayCore* OnesCore( vector<int> shape );

    NArrayCore* OneHotCore( vector<int> integers, int num );

    void Print( NArrayCore* arr );

    void PrintShape( NArrayCore* arr );

    int SizeFromShape( vector<int> shape );

    int MultiToFlatIndex( vector<int> index, vector<int> shape );

    vector<int> FlatToMultiIndex( int index, vector<int> shape );

    vector<int> GetShapeForBroadcast( vector<int> shape1, vector<int> shape2 );
}