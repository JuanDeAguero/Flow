// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#define NARRAY std::shared_ptr<Flow::NArray>

namespace Flow
{
    using namespace std;

    class NArray
    {

    public:

        enum Operation
        {
            NONE,
            ADD, BMM, BROADCAST, CONV1D, CONV2D,
            CROSSENTROPY, EXP, FOLD2D, GATHER, INDEX,
            LOG, MAX, MM, MUL, POW,
            PROD, RELU, RESHAPE, SQUEEZE, SUM,
            TANH, TRANSPOSE, UNFOLD2D, UNSQUEEZE
        };

        NArray( vector<int> shape, const vector<float>& data );

        NArray( vector<int> shape, float* deviceData, vector<NARRAY> operands, Operation op );

        NArray( vector<int> shape );  // Constructor for gradients.

        ~NArray();

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

        float Exponent;

        vector<int> FoldOutShape2d, FoldKernel2d;

        int GatherDim;

        NARRAY GatherIndex;

        int IndexDim;

        NARRAY Index;
        
        int MaxDim;

        int ProdDim;
        
        int SumDim;
        
        int TransposeFirstDim, TransposeSecondDim;

        vector<int> UnfoldKernel2d, UnfoldStride2d;

        vector<NArray*> TopologicalSort();

        void BuildTopo( NArray* current, unordered_set<NArray*>& visited, vector<NArray*>& topo );

        void Backward();

        void BackwardAdd();

        void BackwardBMM();
        
        void BackwardBroadcast();
        
        void BackwardExp();

        void BackwardFold2d();
        
        void BackwardGather();
        
        void BackwardIndex();
        
        void BackwardLog();
        
        void BackwardMax();
        
        void BackwardMul();
        
        void BackwardPow();

        void BackwardProd();
        
        void BackwardReLU();
        
        void BackwardReshape();

        void BackwardSqueeze();
        
        void BackwardSum();
        
        void BackwardTanh();
        
        void BackwardTranspose();

        void BackwardUnfold2d();
        
        void BackwardUnsqueeze();

        friend NARRAY Fold2d( NARRAY arr, vector<int> outShape, vector<int> kernel );

        friend NARRAY Gather( NARRAY arr, int dim, NARRAY index );

        friend NARRAY Index( NARRAY arr, int dim, NARRAY index );

        friend NARRAY Max( NARRAY arr, int dim );

        friend NARRAY Pow( NARRAY arr, float exponent );

        friend NARRAY Prod( NARRAY arr, int dim );
        
        friend NARRAY Sum( NARRAY arr, int dim );

        friend NARRAY Transpose( NARRAY arr, int firstDim, int secondDim );

        friend NARRAY Unfold2d( NARRAY arr, vector<int> kernel, vector<int> stride );
        
    };

    NARRAY Create( vector<int> shape, const vector<float>& data );

    NARRAY Create( vector<int> shape, float* deviceData, vector<NARRAY> operands,
        NArray::Operation op );

    NARRAY Add( NARRAY arr1, NARRAY arr2 );

    NARRAY BMM( NARRAY arr1, NARRAY arr2 );

    NARRAY Broadcast( NARRAY arr, vector<int> shape );

    NARRAY Conv2d( NARRAY arr, NARRAY weight );

    NARRAY CrossEntropy( NARRAY arr1, NARRAY arr2 );

    NARRAY Div( NARRAY arr1, NARRAY arr2 );

    NARRAY Exp( NARRAY arr );

    NARRAY Fold2d( NARRAY arr, vector<int> outShape, vector<int> kernel );

    NARRAY Gather( NARRAY arr, int dim, NARRAY index );

    NARRAY Index( NARRAY arr, int dim, NARRAY index );

    NARRAY Log( NARRAY arr );

    NARRAY Matmul( NARRAY arr1, NARRAY arr2 );

    NARRAY Max( NARRAY arr, int dim );

    NARRAY MaxPool2d( NARRAY arr, vector<int> kernel );

    NARRAY Mean( NARRAY arr, int dim );

    NARRAY MM( NARRAY arr1, NARRAY arr2 );

    NARRAY Mul( NARRAY arr1, NARRAY arr2 );

    NARRAY Mul( NARRAY arr, float literal );

    NARRAY Neg( NARRAY arr );
    
    NARRAY Pow( NARRAY arr, float exponent );

    NARRAY Prod( NARRAY arr, int dim );

    NARRAY ReLU( NARRAY arr );

    NARRAY Reshape( NARRAY arr, vector<int> shape );

    NARRAY Softmax( NARRAY arr, int dim );

    NARRAY Squeeze( NARRAY arr, int dim );

    NARRAY Sub( NARRAY arr1, NARRAY arr2 );

    NARRAY Sum( NARRAY arr, int dim );

    NARRAY Tanh( NARRAY arr );

    NARRAY Transpose( NARRAY arr, int firstDim, int secondDim );

    NARRAY Unfold2d( NARRAY arr, vector<int> kernel, vector<int> stride );
    
    NARRAY Unsqueeze( NARRAY arr, int dim );

    NARRAY Random( vector<int> shape );

    NARRAY RandomPermutation( int n );

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

    vector< pair< NARRAY, NARRAY > > CreateBatches( NARRAY arr1, NARRAY arr2, int batchSize );
}