// Copyright (c) 2023 Juan M. G. de Agüero

#include <cuda_runtime.h>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArray::NArray() { NArray::NArray( new NArrayCore( { 1 }, { 0.0f } ) ); }

Flow::NArray::NArray( NArrayCore* arr ) : Array(arr) {}

Flow::NArray::~NArray() {}

bool Flow::NArray::IsValid()
{
    if ( Array != nullptr ) return true;
    else return false;
}

Flow::NArrayCore* Flow::NArray::GetCore() { return Array; }

float Flow::NArray::Get( vector<int> coordinates ) { return Array->Get(coordinates); }

vector<float> Flow::NArray::Get() { return Array->Get(); }

float* Flow::NArray::GetData() { return Array->GetData(); }

vector<int> Flow::NArray::GetShape() { return Array->GetShape(); }

Flow::NArray Flow::NArray::GetGradient() { return NArray(Array->GetGradient()); }

void Flow::NArray::Set( vector<int> coordinates, float value ) { Array->Set( coordinates, value ); }

void Flow::NArray::Reset( float value ) { Array->Reset(value); }

void Flow::NArray::Backpropagate() { Array->Backpropagate(); }

Flow::NArray Flow::NArray::Copy() { return NArray(Array->Copy()); }

Flow::NArray Flow::Create( vector<int> shape, vector<float> data ) { return NArray( new NArrayCore( shape, data ) ); }

vector<reference_wrapper<Flow::NArray>> Flow::SavedArrays;

void Flow::CleanUp()
{
    if ( GetCUDAFreeMemory() < 5000.0f )
    {
        vector<vector<float>> copyData;
        for ( NArray& arr : SavedArrays ) copyData.push_back(arr.Get());
        cudaDeviceReset();
        for ( int i = 0; i < SavedArrays.size(); i++ )
            SavedArrays[i].get() = Create( SavedArrays[i].get().GetShape(), copyData[i] );
    }
}

void Flow::Save( vector<reference_wrapper<NArray>> savedArrays )
{
    for ( auto arr : savedArrays ) SavedArrays.push_back(arr);
}

Flow::NArray Flow::Add( NArray arr1, NArray arr2 ) { return NArray( Add( arr1.GetCore(), arr2.GetCore() ) ); }

Flow::NArray Flow::Broadcast( NArray arr, vector<int> shape ) { return NArray( Broadcast( arr.GetCore(), shape ) ); }

Flow::NArray Flow::CrossEntropy( NArray arr1, NArray arr2 ) { return CrossEntropy( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArray Flow::Div( NArray arr1, NArray arr2 ) { return NArray( Div( arr1.GetCore(), arr2.GetCore() ) ); }

Flow::NArray Flow::Exp( NArray arr ) { return NArray( Exp(arr.GetCore()) ); }

Flow::NArray Flow::Gather( NArray arr, int dim, NArray index ) { return NArray( Gather( arr.GetCore(), dim, index.GetCore() ) ); }

Flow::NArray Flow::Index( NArray arr, int dim, NArray index ) { return NArray( Index( arr.GetCore(), dim, index.GetCore() ) ); }

Flow::NArray Flow::Log( NArray arr ) { return NArray( Log(arr.GetCore()) ); }

Flow::NArray Flow::Max( NArray arr, int dim ) { return NArray( Max( arr.GetCore(), dim ) ); }

Flow::NArray Flow::Mean( NArray arr, int dim ) { return NArray( Mean( arr.GetCore(), dim ) ); }

Flow::NArray Flow::MM( NArray arr1, NArray arr2 ) { return NArray( MM( arr1.GetCore(), arr2.GetCore() ) ); }

Flow::NArray Flow::Mul( NArray arr, float literal ) { return NArray( Mul( arr.GetCore(), literal ) ); }

Flow::NArray Flow::Mul( NArray arr1, NArray arr2 ) { return NArray( Mul( arr1.GetCore(), arr2.GetCore() ) ); }

Flow::NArray Flow::Neg( NArray arr ) { return NArray( Neg(arr.GetCore()) ); }

Flow::NArray Flow::Pow( NArray arr, float exponent ) { return NArray( Pow( arr.GetCore(), exponent ) ); }

Flow::NArray Flow::ReLU( NArray arr ) { return NArray( ReLU(arr.GetCore()) ); }

Flow::NArray Flow::Reshape( NArray arr, vector<int> shape ) { return NArray( Reshape( arr.GetCore(), shape ) ); }

Flow::NArray Flow::Softmax( NArray arr, int dim ) { return Softmax( arr.GetCore(), dim ); }

Flow::NArray Flow::Sub( NArray arr1, NArray arr2 ) { return NArray( Sub( arr1.GetCore(), arr2.GetCore() ) ); }

Flow::NArray Flow::Sum( NArray arr, int dim ) { return NArray( Sum( arr.GetCore(), dim ) ); }

Flow::NArray Flow::Tanh( NArray arr ) { return NArray( Tanh(arr.GetCore()) ); }

Flow::NArray Flow::Transpose( NArray arr, int firstDim, int secondDim ) { return NArray( Transpose( arr.GetCore(), firstDim, secondDim ) ); }

Flow::NArray Flow::Unsqueeze( NArray arr, int dim ) { return NArray( Unsqueeze( arr.GetCore(), dim ) ); }

Flow::NArray Flow::Random( vector<int> shape ) { return NArray(RandomCore(shape)); }

Flow::NArray Flow::Zeros( vector<int> shape ) { return NArray(ZerosCore(shape)); }

Flow::NArray Flow::Ones( vector<int> shape ) { return NArray(OnesCore(shape)); }

Flow::NArray Flow::OneHot( vector<int> integers, int num ) { return NArray( OneHotCore( integers, num ) ); }

void Flow::Print( NArray arr ) { Print(arr.GetCore()); }