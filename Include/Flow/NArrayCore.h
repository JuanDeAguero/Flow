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

            ADD, MUL,
            MM,
            POW, EXP,
            TANH, RELU, LOG,
            SUM, MAX,
            RESHAPE, TRANSPOSE, BROADCAST,
            GATHER, SQUEEZE, UNSQUEEZE,
            INDEX,

            NEG, SUB, DIV,
            MEAN, SOFTMAX, CROSSENTROPY
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

        void BackwardMul_CUDA();

        void BackwardMM();
        
        void BackwardMM_CUDA();

        void BackwardPow();
        
        void BackwardPow_CUDA();

        void BackwardExp();
        
        void BackwardExp_CUDA();

        void BackwardTanh();
        
        void BackwardTanh_CUDA();

        void BackwardReLU();
        
        void BackwardReLU_CUDA();

        void BackwardLog();
        
        void BackwardLog_CUDA();

        void BackwardSum();
        
        void BackwardSum_CUDA();

        void BackwardMax();
        
        void BackwardMax_CUDA();

        void BackwardReshape();

        void BackwardTranspose();

        void BackwardBroadcast();
        
        void BackwardBroadcast_CUDA();

        void BackwardGather();
        
        void BackwardGather_CUDA();

        void BackwardUnsqueeze();

        void BackwardIndex();
        
        void BackwardIndex_CUDA();

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

    NArrayCore* MM_CUDA( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Pow( NArrayCore* arr, float exponent );

    NArrayCore* Pow_CUDA( NArrayCore* arr, float exponent );

    NArrayCore* Exp( NArrayCore* arr );

    NArrayCore* Exp_CUDA( NArrayCore* arr );

    NArrayCore* Tanh( NArrayCore* arr );

    NArrayCore* Tanh_CUDA( NArrayCore* arr );

    NArrayCore* ReLU( NArrayCore* arr );

    NArrayCore* ReLU_CUDA( NArrayCore* arr );

    NArrayCore* Log( NArrayCore* arr );

    NArrayCore* Log_CUDA( NArrayCore* arr );

    NArrayCore* Sum( NArrayCore* arr, int dim );

    NArrayCore* Sum_CUDA( NArrayCore* arr, int dim );

    NArrayCore* Max( NArrayCore* arr, int dim );

    NArrayCore* Max_CUDA( NArrayCore* arr, int dim );

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape );

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim );

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape );

    NArrayCore* Broadcast_CUDA( NArrayCore* arr, vector<int> shape );

    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Gather_CUDA( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Unsqueeze( NArrayCore* arr, int dim );

    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Index_CUDA( NArrayCore* arr, int dim, NArrayCore* index );

    NArrayCore* Neg( NArrayCore* arr );

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* Mean( NArrayCore* arr, int dim );

    NArrayCore* Softmax( NArrayCore* arr );

    NArrayCore* CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 );

    int SizeFromShape( vector<int> shape );

    int MultiToFlatIndex( vector<int> index, vector<int> shape );

    vector<int> FlatToMultiIndex( int index, vector<int> shape );

    vector<int> GetShapeForBroadcast( NArrayCore* arr1, NArrayCore* arr2 );

    NArrayCore* RandomCore( vector<int> shape );

    NArrayCore* ZerosCore( vector<int> shape );

    NArrayCore* OnesCore( vector<int> shape );

    NArrayCore* OneHotCore( vector<int> integers, int num );

    void Print( NArrayCore* arr );
}