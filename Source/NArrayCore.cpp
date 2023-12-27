// Copyright (c) 2023 Juan M. G. de Agüero

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <cuda_runtime.h>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArrayCore::NArrayCore( vector<int> shape, const vector<float>& data )
    : Shape(shape), Gradient( new NArrayCore(shape) ), Op(Operation::NONE),
      Saved(false), GatherIndex(nullptr), Index(nullptr)
{
    cudaMalloc( (void**)&Data, data.size() * sizeof(float) );
    cudaMemcpy( Data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice );
}

Flow::NArrayCore::NArrayCore( vector<int> shape, float* deviceData, vector<NArrayCore*> operands, Operation op )
    : Data(deviceData), Shape(shape), Gradient( new NArrayCore(shape) ), Operands(operands), Op(op),
      Saved(false), GatherIndex(nullptr), Index(nullptr) {}

Flow::NArrayCore::~NArrayCore() {}

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

Flow::NArrayCore* Flow::NArrayCore::GetGradient() { return Gradient; }

void Flow::NArrayCore::Backpropagate()
{
    if ( Operands.size() == 0 ) return;
    Gradient->Reset(1.0f);
    auto topo = TopologicalSort();
    for ( Flow::NArrayCore* arr : topo ) arr->Backward();
}

Flow::NArrayCore* Flow::NArrayCore::Copy()
{
    int size = SizeFromShape(Shape);
    vector<float> data(size);
    cudaMemcpy( data.data(), Data, size * sizeof(float), cudaMemcpyDeviceToHost );
    return new NArrayCore( Shape, data );
}

void Flow::NArrayCore::Destroy()
{
    cudaFree(Data);
    cudaFree(Gradient->Data);
    if ( Operands.size() > 0 && Operands[0]->Saved == false ) Operands[0]->Destroy();
    if ( Operands.size() == 2 && Operands[1]->Saved == false ) Operands[1]->Destroy();
    if ( GatherIndex && GatherIndex->Saved == false ) GatherIndex->Destroy();
    if ( Index && Index->Saved == false ) Index->Destroy();
    delete Gradient;
}

Flow::NArrayCore::NArrayCore( vector<int> shape )
    : Shape(shape), Gradient(nullptr), Op(Operation::NONE), Saved(false), GatherIndex(nullptr), Index(nullptr)
{
    cudaMalloc( (void**)&Data, SizeFromShape(shape) * sizeof(float) );
    cudaMemset( Data, 0, SizeFromShape(shape) * sizeof(float) );
}

vector<Flow::NArrayCore*> Flow::NArrayCore::TopologicalSort()
{
    unordered_set<NArrayCore*> visited;
    vector<NArrayCore*> topo;
    if (!Data) return topo;
    BuildTopo( this, visited, topo );
    reverse( topo.begin(), topo.end() );
    return topo;
}

void Flow::NArrayCore::BuildTopo( NArrayCore* current, unordered_set<NArrayCore*>& visited, vector<NArrayCore*>& topo )
{
    if ( visited.find(current) != visited.end() || current->Operands.size() == 0 ) return;
    visited.insert(current);
    NArrayCore* first = current->Operands[0];
    if (first)
    {
        BuildTopo( first, visited, topo );
        if ( current->Operands.size() != 1 )
        {
            NArrayCore* second = current->Operands[1];
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
        case Operation::NONE:                           break;
        case Operation::ADD:       BackwardAdd();       break;
        case Operation::BROADCAST: BackwardBroadcast(); break;
        case Operation::EXP:       BackwardExp();       break;
        case Operation::GATHER:    BackwardGather();    break;
        case Operation::INDEX:     BackwardIndex();     break;
        case Operation::LOG:       BackwardLog();       break;
        case Operation::MAX:       BackwardMax();       break;
        case Operation::MM:        BackwardMM();        break;
        case Operation::MUL:       BackwardMul();       break;
        case Operation::POW:       BackwardPow();       break;
        case Operation::RELU:      BackwardReLU();      break;
        case Operation::RESHAPE:   BackwardReshape();   break;
        case Operation::SUM:       BackwardSum();       break;
        case Operation::TANH:      BackwardTanh();      break;
        case Operation::TRANSPOSE: BackwardTranspose(); break;
        case Operation::UNSQUEEZE: BackwardUnsqueeze(); break;
        default: break;
    }
}

Flow::NArrayCore* Flow::Neg( NArrayCore* arr )
{
    return Mul( arr, -1.0f );
}

Flow::NArrayCore* Flow::Sub( NArrayCore* arr1, NArrayCore* arr2 )
{
    return Add( arr1, Neg(arr2) );
}

Flow::NArrayCore* Flow::Div( NArrayCore* arr1, NArrayCore* arr2 )
{
    return Mul( arr1, Pow( arr2, -1.0f ) );
}

Flow::NArrayCore* Flow::Mean( NArrayCore* arr, int dim )
{
    NArrayCore* sum = Sum( arr, dim );
    float numElements = (float)arr->GetShape()[dim];
    NArrayCore* n = new NArrayCore( { 1 }, { numElements } );
    return Div( sum, n );
}

Flow::NArrayCore* Flow::Softmax( NArrayCore* arr, int dim )
{
    return Sub(
        Sub( arr, Max( arr, dim ) ),
        Log( Sum( Exp( Sub( arr, Max( arr, dim ) ) ), dim ) ) );
}

Flow::NArrayCore* Flow::CrossEntropy( NArrayCore* arr1, NArrayCore* arr2 )
{
    return Neg( Mean( Gather( Softmax( arr1, 1 ), 1, Unsqueeze( arr2, 1 ) ), 0 ) );
}

Flow::NArrayCore* Flow::Random( vector<int> shape )
{
    random_device randomDevice;
    mt19937 generator(randomDevice());
    normal_distribution<float> distribution( 0.0f, 1.0f );
    int size = SizeFromShape(shape);
    vector<float> data( size, 0.0f );
    for ( int i = 0; i < size; i++ ) data[i] = distribution(generator);
    return new NArrayCore( shape, data );
}

Flow::NArrayCore* Flow::Zeros( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 0.0f );
    return new NArrayCore( shape, data );
}

Flow::NArrayCore* Flow::Ones( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 1.0f );
    return new NArrayCore( shape, data );
}

Flow::NArrayCore* Flow::OneHot( vector<int> integers, int num )
{
    vector<float> data( integers.size() * num );
    NArrayCore* arr = new NArrayCore( { (int)integers.size(), num }, data );
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

void Flow::Print( NArrayCore* arr )
{
    int size = SizeFromShape(arr->GetShape());
    vector<float> data(size);
    cudaMemcpy( data.data(), arr->GetData(), size * sizeof(float), cudaMemcpyDeviceToHost );
    for ( float value : data ) Print(value);
}

void Flow::PrintShape( NArrayCore* arr )
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
    cudaError_t cuda_status = cudaMemGetInfo( &freeByte, &totalByte );
    return freeByte / 1024.0f / 1024.0f;
}