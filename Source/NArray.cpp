// Copyright (c) 2023 Juan M. G. de Agüero

#include <cuda_runtime.h>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArray::NArray() { NArray::NArray( new NArrayCore( { 1 }, { 0.0f } ) ); }

Flow::NArray::NArray( NArrayCore* arr ) : Array(arr) { Array->Saved = true; }

Flow::NArray::~NArray() { Array->Destroy(); delete Array; }

Flow::NArrayCore* Flow::NArray::GetCore() { return Array; }

float Flow::NArray::Get( vector<int> coordinates ) { return Array->Get(coordinates); }

vector<float> Flow::NArray::Get() { return Array->Get(); }

float* Flow::NArray::GetData() { return Array->GetData(); }

vector<int> Flow::NArray::GetShape() { return Array->GetShape(); }

Flow::NArrayCore* Flow::NArray::GetGradient() { return Array->GetGradient(); }

void Flow::NArray::Set( vector<int> coordinates, float value ) { Array->Set( coordinates, value ); }

void Flow::NArray::Reset( float value ) { Array->Reset(value); }

void Flow::NArray::Backpropagate() { Array->Backpropagate(); }

Flow::NArrayCore* Flow::NArray::Copy() { return Array->Copy(); }

void Flow::NArray::Assign( NArrayCore* arr ) { Array->Destroy(); Array = arr; Array->Saved = true; }

Flow::NArrayCore* Flow::Create( vector<int> shape, const vector<float>& data ) { return new NArrayCore( shape, data ); }

Flow::NArrayCore* Flow::Add( NArray& arr1, NArray& arr2 ) { return Add( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Add( NArrayCore* arr1, NArray& arr2 ) { return Add( arr1, arr2.GetCore() ); }

Flow::NArrayCore* Flow::Broadcast( NArray& arr, vector<int> shape ) { return Broadcast( arr.GetCore(), shape ); }

Flow::NArrayCore* Flow::CrossEntropy( NArray& arr1, NArray& arr2 ) { return CrossEntropy( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Div( NArray& arr1, NArray& arr2 ) { return Div( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Exp( NArray& arr ) { return Exp(arr.GetCore()); }

Flow::NArrayCore* Flow::Gather( NArray& arr, int dim, NArray& index ) { return Gather( arr.GetCore(), dim, index.GetCore() ); }

Flow::NArrayCore* Flow::Index( NArray& arr, int dim, NArray& index ) { return Index( arr.GetCore(), dim, index.GetCore() ); }

Flow::NArrayCore* Flow::Log( NArray& arr ) { return Log(arr.GetCore()); }

Flow::NArrayCore* Flow::Max( NArray& arr, int dim ) { return Max( arr.GetCore(), dim ); }

Flow::NArrayCore* Flow::Mean( NArray& arr, int dim ) { return Mean( arr.GetCore(), dim ); }

Flow::NArrayCore* Flow::MM( NArray& arr1, NArray& arr2 ) { return MM( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Mul( NArray& arr, float literal ) { return Mul( arr.GetCore(), literal ); }

Flow::NArrayCore* Flow::Mul( NArray& arr1, NArray& arr2 ) { return Mul( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Neg( NArray& arr ) { return Neg(arr.GetCore() ); }

Flow::NArrayCore* Flow::Pow( NArray& arr, float exponent ) { return Pow( arr.GetCore(), exponent ); }

Flow::NArrayCore* Flow::ReLU( NArray& arr ) { return ReLU(arr.GetCore()); }

Flow::NArrayCore* Flow::Reshape( NArray& arr, vector<int> shape ) { return Reshape( arr.GetCore(), shape ); }

Flow::NArrayCore* Flow::Softmax( NArray& arr, int dim ) { return Softmax( arr.GetCore(), dim ); }

Flow::NArrayCore* Flow::Sub( NArray& arr1, NArray& arr2 ) { return Sub( arr1.GetCore(), arr2.GetCore() ); }

Flow::NArrayCore* Flow::Sum( NArray& arr, int dim ) { return Sum( arr.GetCore(), dim ); }

Flow::NArrayCore* Flow::Tanh( NArray& arr ) { return Tanh(arr.GetCore()); }

Flow::NArrayCore* Flow::Transpose( NArray& arr, int firstDim, int secondDim ) { return Transpose( arr.GetCore(), firstDim, secondDim ); }

Flow::NArrayCore* Flow::Unsqueeze( NArray& arr, int dim ) { return Unsqueeze( arr.GetCore(), dim ); }

void Flow::Print( NArray& arr ) { Print(arr.GetCore()); }