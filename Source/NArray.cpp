// Copyright (c) 2023 Juan M. G. de Agüero

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <cuda_runtime.h>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArrayCore::NArrayCore( vector<int> shape, const vector<float>& data )
    : Shape(shape), Gradient( make_shared<NArrayCore>(shape) ), Op(NArray::Operation::NONE),
      Saved(false), GatherIndex(nullptr), Index(nullptr)
{
    cudaMalloc( (void**)&Data, data.size() * sizeof(float) );
    cudaMemcpy( Data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice );
}

Flow::NArrayCore::NArrayCore( vector<int> shape, float* deviceData, vector<NARRAY> operands,
    NArray::Operation op )
    : Data(deviceData), Shape(shape), Gradient( make_shared<NArrayCore>(shape) ),
      Operands(operands), Op(op), Saved(false), GatherIndex(nullptr), Index(nullptr) {}

Flow::NArrayCore::NArrayCore( vector<int> shape )
    : Shape(shape), Gradient(nullptr), Op(NArray::Operation::NONE), Saved(false),
      GatherIndex(nullptr), Index(nullptr)
{
    cudaMalloc( (void**)&Data, SizeFromShape(shape) * sizeof(float) );
    cudaMemset( Data, 0, SizeFromShape(shape) * sizeof(float) );
}

Flow::NArrayCore::~NArrayCore()
{
    cudaFree(Data);
}

float Flow::NArrayCore::Get( vector<int> coordinates )
{
    int index = MultiToFlatIndex( coordinates, Shape );
    if ( index >= 0 && index < SizeFromShape(Shape) )
    {
        float value;
        cudaMemcpy( &value, &Data[index], sizeof(float), cudaMemcpyDeviceToHost );
        return value;
    }
}

vector<float> Flow::NArrayCore::Get()
{
    int size = SizeFromShape(Shape);
    vector<float> data(size);
    cudaMemcpy( data.data(), Data, size * sizeof(float), cudaMemcpyDeviceToHost );
    return data;
}

float* Flow::NArrayCore::GetData() { return Data; }

vector<int> Flow::NArrayCore::GetShape() { return Shape; }

int* Flow::NArrayCore::GetShapeData() { return Shape.data(); }

NARRAY Flow::NArrayCore::GetGradient() { return Gradient; }

void Flow::NArrayCore::Backpropagate()
{
    if ( Operands.size() == 0 ) return;
    Gradient->Reset(1.0f);
    auto topo = TopologicalSort();
    for ( Flow::NArrayCore* arr : topo ) arr->Backward();
}

NARRAY Flow::NArrayCore::Copy()
{
    int size = SizeFromShape(Shape);
    vector<float> data(size);
    cudaMemcpy( data.data(), Data, size * sizeof(float), cudaMemcpyDeviceToHost );
    return make_shared<NArrayCore>( Shape, data );
}

void Flow::NArrayCore::Copy( NARRAY arr )
{
    int size = SizeFromShape(Shape);
    cudaMemcpy( Data, arr->Data, size * sizeof(float), cudaMemcpyDeviceToDevice );
}

vector<Flow::NArrayCore*> Flow::NArrayCore::TopologicalSort()
{
    unordered_set<NArrayCore*> visited;
    vector<NArrayCore*> topo;
    BuildTopo( this, visited, topo );
    reverse( topo.begin(), topo.end() );
    return topo;
}

void Flow::NArrayCore::BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited,
    vector<NArrayCore*>& topo )
{
    if ( visited.find(current) != visited.end() || current->Operands.size() == 0 ) return;
    visited.insert(current);
    NArrayCore* first = current->Operands[0].get();
    if (first)
    {
        BuildTopo( first, visited, topo );
        if ( current->Operands.size() != 1 )
        {
            NArrayCore* second = current->Operands[1].get();
            if (second) BuildTopo( second, visited, topo );
        }
    }
    topo.push_back(current);
}

void Flow::NArrayCore::Backward()
{
    if ( Operands.size() == 0 ) return;
    switch (Op)
    {
        case NArray::Operation::NONE:                           break;
        case NArray::Operation::ADD:       BackwardAdd();       break;
        case NArray::Operation::BROADCAST: BackwardBroadcast(); break;
        case NArray::Operation::EXP:       BackwardExp();       break;
        case NArray::Operation::GATHER:    BackwardGather();    break;
        case NArray::Operation::INDEX:     BackwardIndex();     break;
        case NArray::Operation::LOG:       BackwardLog();       break;
        case NArray::Operation::MAX:       BackwardMax();       break;
        case NArray::Operation::MM:        BackwardMM();        break;
        case NArray::Operation::MUL:       BackwardMul();       break;
        case NArray::Operation::POW:       BackwardPow();       break;
        case NArray::Operation::RELU:      BackwardReLU();      break;
        case NArray::Operation::RESHAPE:   BackwardReshape();   break;
        case NArray::Operation::SUM:       BackwardSum();       break;
        case NArray::Operation::TANH:      BackwardTanh();      break;
        case NArray::Operation::TRANSPOSE: BackwardTranspose(); break;
        case NArray::Operation::UNSQUEEZE: BackwardUnsqueeze(); break;
        default: break;
    }
}

NARRAY Flow::NArray::Create( vector<int> shape, const vector<float>& data )
{
    return make_shared<NArrayCore>( shape, data );
}

NARRAY Flow::NArray::Create( vector<int> shape, float* deviceData, vector<NARRAY> operands,
    NArray::Operation op )
{
    return make_shared<NArrayCore>( shape, deviceData, operands, op );
}

NARRAY Flow::Neg( NARRAY arr )
{
    return Mul( arr, -1.0f );
}

NARRAY Flow::Sub( NARRAY arr1, NARRAY arr2 )
{
    return Add( arr1, Neg(arr2) );
}

NARRAY Flow::Div( NARRAY arr1, NARRAY arr2 )
{
    return Mul( arr1, Pow( arr2, -1.0f ) );
}

NARRAY Flow::Mean( NARRAY arr, int dim )
{
    NARRAY sum = Sum( arr, dim );
    float numElements = (float)arr->GetShape()[dim];
    NARRAY n = NArray::Create( { 1 }, { numElements } );
    return Div( sum, n );
}

NARRAY Flow::Softmax( NARRAY arr, int dim )
{
    NARRAY index = NArray::Create( { 1 }, { 0.0f } );
    NARRAY exp = Exp( Sub( arr, Index( Max( arr, dim ), dim, index ) ) );
    return Div( exp, Sum( exp, dim ) );
}

NARRAY Flow::CrossEntropy( NARRAY arr1, NARRAY arr2 )
{
    NARRAY small = NArray::Create( { 1 }, { 1e-10f } );
    NARRAY arrUnsqueezed2 = Unsqueeze( arr2, 1 );
    return Mean( Neg( Log( Add( Gather( Softmax( arr1, 1 ), 1, arrUnsqueezed2 ), small ) ) ), 0 );
}

NARRAY Flow::Random( vector<int> shape )
{
    random_device randomDevice;
    mt19937 generator(randomDevice());
    normal_distribution<float> distribution( 0.0f, 1.0f );
    int size = SizeFromShape(shape);
    vector<float> data( size, 0.0f );
    for ( int i = 0; i < size; i++ ) data[i] = distribution(generator);
    return NArray::Create( shape, data );
}

NARRAY Flow::Zeros( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 0.0f );
    return NArray::Create( shape, data );
}

NARRAY Flow::Ones( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 1.0f );
    return NArray::Create( shape, data );
}

NARRAY Flow::OneHot( vector<int> integers, int num )
{
    vector<float> data( integers.size() * num );
    NARRAY arr = NArray::Create( { (int)integers.size(), num }, data );
    for ( int i = 0; i < integers.size(); i++ )
    {
        for ( int j = 0; j < num; j++ )
        {
            float value = 0.0f;
            if ( integers[i] == j ) value = 1.0f;
            arr->Set( { i, j }, value );
        }
    }
    return arr;
}

void Flow::Print( NARRAY arr )
{
    int size = SizeFromShape(arr->GetShape());
    vector<float> data(size);
    cudaMemcpy( data.data(), arr->GetData(), size * sizeof(float), cudaMemcpyDeviceToHost );
    for ( float value : data ) Print(value);
}

void Flow::PrintShape( NARRAY arr )
{
    for ( int value : arr->GetShape() ) Print((float)value);
}

int Flow::SizeFromShape( vector<int> shape )
{
    int size = shape[0];
    for ( int i = 1; i < shape.size(); i++ ) size *= shape[i];
    return size;
}

int Flow::MultiToFlatIndex( vector<int> index, vector<int> shape )
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

vector<int> Flow::FlatToMultiIndex( int index, vector<int> shape )
{
    vector<int> multiIndex(shape.size());
    for ( int i = shape.size() - 1; i >= 0; i-- )
    {
        multiIndex[i] = index % shape[i];
        index /= shape[i];
    }
    return multiIndex;
}

float Flow::GetCUDAFreeMemory()
{
    size_t freeByte;
    size_t totalByte;
    cudaError_t cudaStatus = cudaMemGetInfo( &freeByte, &totalByte );
    return freeByte / 1024.0f / 1024.0f;
}