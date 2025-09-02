// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArray::NArray( const vector<int>& shape, const vector<float>& data )
    : Shape(shape), MetaParent(nullptr), Stride(StrideFromShape(Shape)), StorageOffset(0),
      Gradient(nullptr), Op(NArray::Operation::NONE)
{
    CUDA_AllocateFloat( Data, data );
    CUDA_AllocateInt( ShapeDevice, Shape );
    CUDA_AllocateInt( StrideDevice, Stride );
    CreateDeviceStruct();
}

Flow::NArray::NArray( const vector<int>& shape )
    : Shape(shape), MetaParent(nullptr), Stride(StrideFromShape(Shape)), StorageOffset(0),
      Gradient(nullptr), Op(NArray::Operation::NONE)
{
    cudaMalloc( (void**)&Data, SizeFromShape(Shape) * sizeof(float) );
    Reset(0.0f);
    CUDA_AllocateInt( ShapeDevice, Shape );
    CUDA_AllocateInt( StrideDevice, Stride );
    CreateDeviceStruct();
}

Flow::NArray::NArray( vector<int> shape, vector<NARRAY> operands, NArray::Operation op )
    : Shape(shape), MetaParent(nullptr), Stride(StrideFromShape(Shape)), StorageOffset(0),
      Gradient(nullptr), Operands(operands), Op(op)
{
    cudaMalloc( (void**)&Data, SizeFromShape(Shape) * sizeof(float) );
    Reset(0.0f);
    CUDA_AllocateInt( ShapeDevice, Shape );
    CUDA_AllocateInt( StrideDevice, Stride );
    CreateDeviceStruct();
}

Flow::NArray::NArray( vector<int> shape, vector<int> stride, int storageOffset, NARRAY metaParent,
    vector<NARRAY> operands, Operation op )
    : Data(metaParent->Data), Shape(shape), MetaParent(metaParent), Stride(stride),
      StorageOffset(storageOffset), Gradient(nullptr), Operands(operands), Op(op)
{
    CUDA_AllocateInt( ShapeDevice, Shape );
    CUDA_AllocateInt( StrideDevice, Stride );
    CreateDeviceStruct();
}

Flow::NArray::~NArray()
{
    if (!MetaParent) cudaFree(Data);
    cudaFree(ShapeDevice);
    cudaFree(StrideDevice);
    cudaFree(DeviceStruct);

    auto reset = [&]( NArray* arr, auto&& reset_ref ) -> void
    {
        arr->Gradient.reset();
        for ( auto& operand : arr->Operands )
        {
            if (operand)
                reset_ref( operand.get(), reset_ref );
        }
    };
    reset(this, reset);
}

float Flow::NArray::Get( vector<int> coordinates )
{
    int index = MultiToFlatIndex( coordinates, Stride, StorageOffset );
    if ( index >= 0 && index < SizeFromShape(Shape) )
    {
        float value;
        cudaMemcpy( &value, &Data[index], sizeof(float), cudaMemcpyDeviceToHost );
        return value;
    }
}

vector<float> Flow::NArray::Get()
{
    int totalSize = SizeFromShape(Shape);
    vector<float> data(totalSize);
    if ( Stride == StrideFromShape(Shape) && StorageOffset == 0 )
        cudaMemcpy( data.data(), Data, totalSize * sizeof(float), cudaMemcpyDeviceToHost );
    else {
        for ( int i = 0; i < totalSize; i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, Shape );
            int flatIndex = MultiToFlatIndex( multiIndex, Stride, StorageOffset );
            cudaMemcpy( &data[i], &Data[flatIndex], sizeof(float), cudaMemcpyDeviceToHost );
        }
    }
    return data;
}

float* Flow::NArray::GetData() { return Data; }

vector<int> Flow::NArray::GetShape() { return Shape; }

int* Flow::NArray::GetShapeDevice() { return ShapeDevice; }

vector<int> Flow::NArray::GetStride() { return Stride; }

int* Flow::NArray::GetStrideDevice() { return StrideDevice; }

int Flow::NArray::GetOffset() { return StorageOffset; }

NARRAY Flow::NArray::GetGradient() { return Gradient; }

Flow::NArrayDevice* Flow::NArray::GetDeviceStruct() { return DeviceStruct; }

NARRAY Flow::NArray::Copy()
{
    int size = SizeFromShape(Shape);
    vector<float> data(size);
    cudaMemcpy( data.data(), Data, size * sizeof(float), cudaMemcpyDeviceToHost );
    return make_shared<NArray>( Shape, data );
}

void Flow::NArray::Copy( NARRAY arr )
{
    int size = SizeFromShape(Shape);
    cudaMemcpy( Data, arr->Data, size * sizeof(float), cudaMemcpyDeviceToDevice );
}

void Flow::NArray::Backpropagate()
{
    if (!Gradient)
    {
        Gradient = make_shared<NArray>(Shape);
        Gradient->Reset(1.0f);
    }
    if ( Operands.size() == 0 ) return;
    auto topo = TopologicalSort();
    for ( Flow::NArray* arr : topo ) arr->Backward();
}

void Flow::NArray::Backward()
{
    if ( Operands.size() == 0 ) return;
    for ( int i = 0; i < Operands.size(); i++ )
    {
        if (Operands[i]->Gradient) continue;
        //if ( Op != Operation::RESHAPE && Op != Operation::SQUEEZE && Op != Operation::TRANSPOSE &&
        //    Op != Operation::UNSQUEEZE && Op != Operation::BMM )
        Operands[i]->Gradient = make_shared<NArray>(Operands[i]->Shape);
    }
    switch (Op)
    {
        case NArray::Operation::NONE:                           break;
        case NArray::Operation::ADD:       BackwardAdd();       break;
        case NArray::Operation::BMM:       BackwardBMM();       break;
        case NArray::Operation::BROADCAST: BackwardBroadcast(); break;
        case NArray::Operation::EXP:       BackwardExp();       break;
        case NArray::Operation::FOLD2D:    BackwardFold2d();    break;
        case NArray::Operation::GATHER:    BackwardGather();    break;
        case NArray::Operation::INDEX:     BackwardIndex();     break;
        case NArray::Operation::LOG:       BackwardLog();       break;
        case NArray::Operation::MAX:       BackwardMax();       break;
        case NArray::Operation::MUL:       BackwardMul();       break;
        case NArray::Operation::POW:       BackwardPow();       break;
        case NArray::Operation::PROD:      BackwardProd();      break;
        case NArray::Operation::RELU:      BackwardReLU();      break;
        case NArray::Operation::RESHAPE:   BackwardReshape();   break;
        case NArray::Operation::SQUEEZE:   BackwardSqueeze();   break;
        case NArray::Operation::SUM:       BackwardSum();       break;
        case NArray::Operation::TANH:      BackwardTanh();      break;
        case NArray::Operation::TRANSPOSE: BackwardTranspose(); break;
        case NArray::Operation::UNFOLD2D:  BackwardUnfold2d();  break;
        case NArray::Operation::UNSQUEEZE: BackwardUnsqueeze(); break;
        default: break;
    }
}

vector<Flow::NArray*> Flow::NArray::TopologicalSort()
{
    unordered_set<NArray*> visited;
    vector<NArray*> topo;
    BuildTopo( this, visited, topo );
    reverse( topo.begin(), topo.end() );
    return topo;
}

void Flow::NArray::BuildTopo( NArray* current, unordered_set<NArray*>& visited,
    vector<NArray*>& topo )
{
    if ( visited.find(current) != visited.end() || current->Operands.size() == 0 ) return;
    visited.insert(current);
    NArray* first = current->Operands[0].get();
    if (first)
    {
        BuildTopo( first, visited, topo );
        if ( current->Operands.size() != 1 )
        {
            NArray* second = current->Operands[1].get();
            if (second) BuildTopo( second, visited, topo );
        }
    }
    topo.push_back(current);
}

void Flow::NArray::CreateDeviceStruct()
{
    NArrayDevice deviceStruct( Data, ShapeDevice, Shape.size(), StrideDevice, StorageOffset );
    NArrayDevice* deviceStruct_d;
    cudaMalloc( &deviceStruct_d, sizeof(NArrayDevice) );
    cudaMemcpy( deviceStruct_d, &deviceStruct, sizeof(NArrayDevice), cudaMemcpyHostToDevice );
    DeviceStruct = deviceStruct_d;
}

NARRAY Flow::Create( const vector<int>& shape, const vector<float>& data )
{
    return make_shared<NArray>( shape, data );
}

NARRAY Flow::Random( vector<int> shape )
{
    random_device randomDevice;
    mt19937 generator(randomDevice());
    normal_distribution<float> distribution( 0.0f, 0.01f );
    int size = SizeFromShape(shape);
    vector<float> data( size, 0.0f );
    for ( int i = 0; i < size; i++ ) data[i] = distribution(generator);
    return Create( shape, data );
}

NARRAY Flow::RandomUniform( vector<int> shape, float min, float max )
{
    random_device randomDevice;
    mt19937 generator(randomDevice());
    uniform_real_distribution<float> distribution( min, max );
    int size = SizeFromShape(shape);
    vector<float> data(size);
    for ( int i = 0; i < size; i++ ) data[i] = distribution(generator);
    return Create( shape, data );
}

NARRAY Flow::RandomPermutation( int n )
{
    vector<float> data(n);
    for( int i = 0; i < n; i++ ) data[i] = static_cast<float>(i);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle( data.begin(), data.end(), default_random_engine(seed) );
    return Create( { n }, data );
}

NARRAY Flow::Zeros( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 0.0f );
    return Create( shape, data );
}

NARRAY Flow::Ones( vector<int> shape )
{
    vector<float> data( SizeFromShape(shape), 1.0f );
    return Create( shape, data );
}

void Flow::Print( NARRAY arr )
{
    int size = SizeFromShape(arr->GetShape());
    vector<float> data(size);
    cudaMemcpy( data.data(), arr->GetData(), size * sizeof(float), cudaMemcpyDeviceToHost );
    for ( float value : data ) Print(value);
}

int Flow::SizeFromShape( vector<int> shape )
{
    int size = shape[0];
    for ( int i = 1; i < shape.size(); i++ ) size *= shape[i];
    return size;
}

vector<int> Flow::StrideFromShape( vector<int> shape )
{
    vector<int> strides(shape.size());
    int stride = 1;
    for ( int i = shape.size() - 1; i >= 0; i-- )
    {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

vector<int> Flow::FlatToMultiIndex( int flatIndex, vector<int> shape )
{
    vector<int> multiIndex( shape.size(), 0 );
    int product = 1;
    for ( int i = shape.size() - 1; i >= 0; i-- ) product *= shape[i];
    for ( int i = 0; i < shape.size(); i++ )
    {
        product /= shape[i];
        multiIndex[i] = ( flatIndex / product ) % shape[i];
    }
    return multiIndex;
}

int Flow::MultiToFlatIndex( vector<int> multiIndex, vector<int> stride, int offset )
{
    int flatIndex = offset;
    for ( int i = 0; i < multiIndex.size(); i++ ) flatIndex += multiIndex[i] * stride[i];
    return flatIndex;
}

vector< pair< NARRAY, NARRAY > > Flow::CreateBatches( NARRAY arr1, NARRAY arr2, int batchSize )
{
    vector< pair< NARRAY, NARRAY > > batches;
    int numSamples = arr1->GetShape()[0];
    NARRAY indices = RandomPermutation(numSamples);
    int totalBatches = ( numSamples + batchSize - 1 ) / batchSize;
    for ( int i = 0; i < totalBatches; i++ )
    {
        int start = i * batchSize;
        int length = min( batchSize, numSamples - start );
        NARRAY batchIndices = Zeros({ length });
        for ( int j = 0; j < length; j++ )
            batchIndices->Set( { j }, indices->Get({ start + j }) );
        NARRAY batchX = Index( arr1, 0, batchIndices );
        NARRAY batchY = Index( arr2, 0, batchIndices );
        pair< NARRAY, NARRAY > batchPair = { batchX, batchY };
        batches.push_back(batchPair);
    }
    return batches;
}

NARRAY Flow::FindMetaParent( NARRAY arr )
{
    if (!arr->MetaParent) return arr;
    else return arr->MetaParent;
}

Flow::NArrayDevice::NArrayDevice( float* data, int* shape, int shapeSize, int* stride, int offset )
    : Data(data), Shape(shape), ShapeSize(shapeSize), Stride(stride), Offset(offset) {}

bool Flow::CUDA_AllocateFloat( float*& deviceData, const vector<float>& data )
{
    size_t bytes = data.size() * sizeof(float);
    cudaError_t status = cudaMalloc( (void**)&deviceData, bytes );
    if ( status != cudaSuccess ) return false;
    status = cudaMemcpy( deviceData, data.data(), bytes, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) return false;
    else return true;
}

bool Flow::CUDA_AllocateInt( int*& deviceData, const const vector<int>& data )
{
    size_t bytes = data.size() * sizeof(int);
    cudaError_t status = cudaMalloc( (void**)&deviceData, bytes );
    if ( status != cudaSuccess ) return false;
    status = cudaMemcpy( deviceData, data.data(), bytes, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) return false;
    else return true;
}

float Flow::CUDA_GetFreeMemory()
{
    size_t freeByte;
    size_t totalByte;
    cudaError_t cudaStatus = cudaMemGetInfo( &freeByte, &totalByte );
    return freeByte / 1024.0f / 1024.0f;
}

void Flow::CUDA_DeviceSynchronize()
{
    cudaDeviceSynchronize();
}