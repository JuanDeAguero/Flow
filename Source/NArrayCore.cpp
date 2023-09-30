// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data )
{
    Data = data;
    Shape = shape;
    ComputeStride();
    vector<float> gradientData( SizeFromShape(shape), 0.0f );
    Gradient = new NArrayCore( shape, gradientData, true );
    Op = Operation::NONE;
}

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data, vector<NArrayCore*> operands, Operation op )
{
    Data = data;
    Shape = shape;
    ComputeStride();
    vector<float> gradientData( SizeFromShape(shape), 0.0f );
    Gradient = new NArrayCore( shape, gradientData, true );
    Operands = operands;
    Op = op;
}

float Flow::NArrayCore::Get( vector<int> coordinates )
{
    int index = GetIndex(coordinates);
    if ( index >= 0 && index < Data.size() )
        return Data[index];
    return 0.0f;
}

vector<float> Flow::NArrayCore::Get()
{
    return Data;
}

int Flow::NArrayCore::GetIndex( vector<int> coordinates )
{
    if ( coordinates.size() != Shape.size() )
        return -1;
    int index = 0;
    for ( int i = 0; i < coordinates.size(); i++ )
    {
        if ( coordinates[i] >= Shape[i] || coordinates[i] < 0 )
            return -1;
        index += coordinates[i] * Stride[i];
    }
    return index;
}

vector<int> Flow::NArrayCore::GetShape()
{
    return Shape;
}

vector<int> Flow::NArrayCore::GetStride()
{
    return Stride;
}

Flow::NArrayCore* Flow::NArrayCore::GetGradient()
{
    return Gradient->Copy();
}

void Flow::NArrayCore::Set( vector<int> coordinates, float value )
{
    int index = GetIndex(coordinates);
    if ( index >= 0 && index < Data.size() )
        Data[index] = value;
}

void Flow::NArrayCore::Reset( float value )
{
    for ( int i = 0; i < Data.size(); i++ )
        Data[i] = value;
}

void Flow::NArrayCore::Backpropagate()
{
    Gradient->Reset(1.0f);
    TopologicalSort();
    for ( Flow::NArrayCore* arr : TopologicalSort() )
        arr->Backward();
}

Flow::NArrayCore* Flow::NArrayCore::Copy()
{
    NArrayCore* copy = new NArrayCore( Shape, Data );
    return copy;
}

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data, bool isGradient )
{
    Data = data;
    Shape = shape;
    ComputeStride();
    if (!isGradient)
    {
        vector<float> gradientData( SizeFromShape(shape), 0.0f );
        Gradient = new NArrayCore( shape, gradientData, true );
    }
    Op = Operation::NONE;
}

void Flow::NArrayCore::ComputeStride()
{
    Stride.resize(Shape.size());
    int strideValue = 1;
    for ( int i = Shape.size() - 1; i >= 0; i-- )
    {
        Stride[i] = strideValue;
        strideValue *= Shape[i];
    }
}

vector<Flow::NArrayCore*> Flow::NArrayCore::TopologicalSort()
{
    unordered_set<NArrayCore*> visited;
    vector<NArrayCore*> topo;
    BuildTopo( this, visited, topo );
    reverse( topo.begin(), topo.end() );
    return topo;
}

void Flow::NArrayCore::BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo )
{
    if ( visited.find(current) != visited.end() || current->Operands.size() == 0 )
        return;
    visited.insert(current);
    NArrayCore* first = current->Operands[0];
    if (first) BuildTopo( first, visited, topo );
    if ( current->Operands.size() != 1 )
    {
        NArrayCore* second = current->Operands[1];
        if (second) BuildTopo( second, visited, topo );
    }
    topo.push_back(current);
}

void Flow::NArrayCore::Backward()
{
    if ( Operands.size() == 0 )
        return;
    switch (Op)
    {
        case Operation::NONE:                           break;
        case Operation::ADD:       BackwardAdd();       break;
        case Operation::MUL:       BackwardMul();       break;
        case Operation::MM:        BackwardMM();        break;
        case Operation::POW:       BackwardPow();       break;
        case Operation::EXP:       BackwardExp();       break;
        case Operation::TANH:      BackwardTanh();      break;
        case Operation::RELU:      BackwardReLU();      break;
        case Operation::LOG:       BackwardLog();       break;
        case Operation::SUM:       BackwardSum();       break;
        case Operation::MAX:       BackwardMax();       break;
        case Operation::RESHAPE:   BackwardReshape();   break;
        case Operation::TRANSPOSE: BackwardTranspose(); break;
        case Operation::BROADCAST: BackwardBroadcast(); break;
        case Operation::GATHER:    BackwardGather();    break;
        case Operation::UNSQUEEZE: BackwardUnsqueeze(); break;
        case Operation::INDEX:     BackwardIndex();     break;
    }
}

namespace Flow
{
    NArrayCore* Neg( NArrayCore* arr )
    {
        return Mul( arr, -1.0f );
    }

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return Add( arr1, Neg(arr2) );
    }

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return Mul( arr1, Pow( arr2, -1.0f ) );
    }

    NArrayCore* Mean( NArrayCore* arr )
    {
        NArrayCore* sum = Sum( arr, 0 );
        float numElements = static_cast<float>( SizeFromShape(arr->GetShape()) );
        NArrayCore* n = new NArrayCore( { 1 }, { numElements } );
        return Div( sum, n );
    }

    NArrayCore* Softmax( NArrayCore* arr )
    {
        NArrayCore* index = new NArrayCore( { 1 }, { 0 } );
        NArrayCore* max = Index( Max( arr, 1 ), 1, index );
        return Sub( arr, Sub( max, Log( Sum( Exp( Sub( arr, max ) ), 1 ) ) ) );
    }

    NArrayCore* CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return Neg( Mean( Gather( Softmax(arr1), 1, Unsqueeze( arr2, 1 ) ) ) );
    }

    int SizeFromShape( vector<int> shape )
    {
        int size = shape[0];
        for ( int i = 1; i < shape.size(); i++ )
            size *= shape[i];
        return size;
    }

    NArrayCore* RandomCore( vector<int> shape )
    {
        random_device randomDevice;
        mt19937 generator(randomDevice());
        uniform_real_distribution<float> distribution( -1.0f, 1.0f );
        int size = SizeFromShape(shape);
        vector<float> data( size, 0.0f );
        for ( int i = 0; i < size; i++ )
            data[i] = distribution(generator);
        return new NArrayCore( shape, data );
    }

    void Print( NArrayCore* arr )
    {
        for ( float value : arr->Get() )
            Print(value);
    }

    vector<int> FlatToMultiIndex( int index, vector<int> shape )
    {
        vector<int> multiIndex(shape.size());
        for ( int i = shape.size() - 1; i >= 0; i-- )
        {
            multiIndex[i] = index % shape[i];
            index /= shape[i];
        }
        return multiIndex;
    }

    int MultiToFlatIndex( vector<int> index, vector<int> shape )
    {
        int flatIndex = 0;
        int stride = 1;
        for ( int i = shape.size() - 1; i >= 0; i-- )
        {
            flatIndex += index[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }
}