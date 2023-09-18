// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "NArray.h"
#include "ElementWise.hpp"
#include "Log.h"

#define FLOW_TORCH_MODE

#ifdef FLOW_TORCH_MODE
#include "Torch.hpp"
#endif

using namespace std;

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data )
{
    Data = data;
    Shape = shape;
    vector<float> gradientData( SizeFromShape(shape), 0.0f );
    Gradient = new NArrayCore( shape, gradientData, true );
    Op = Operation::NONE;
}

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data, bool isGradient )
{
    Data = data;
    Shape = shape;
    if (!isGradient)
    {
        vector<float> gradientData( SizeFromShape(shape), 0.0f );
        Gradient = new NArrayCore( shape, gradientData, true );
    }
    Op = Operation::NONE;
}

Flow::NArrayCore::NArrayCore( vector<int> shape, vector<float> data, vector<NArrayCore*> operands, Operation op )
{
    Data = data;
    Shape = shape;
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

vector<int> Flow::NArrayCore::GetShape()
{
    return Shape;
}

int Flow::NArrayCore::GetIndex( vector<int> coordinates )
{
    if ( coordinates.size() != Shape.size() )
        return -1;
    int index = 0;
    int stride = 1;
    for ( int i = coordinates.size() - 1; i >= 0; i-- )
    {
        if ( coordinates[i] >= Shape[i] || coordinates[i] < 0 )
            return -1;
        index += coordinates[i] * stride;
        stride *= Shape[i];
    }
    return index;
}

Flow::NArrayCore* Flow::NArrayCore::GetGradient()
{
    return Gradient;
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

void Flow::NArrayCore::Reshape( vector<int> shape )
{
    Shape = shape;
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

int Flow::NArrayCore::SizeFromShape( vector<int> shape )
{
    int size = shape[0];
    for ( int i = 1; i < shape.size(); i++ )
        size *= shape[i];
    return size;
}

void Flow::NArrayCore::Backward()
{
    if ( Operands.size() == 0 )
        return;
    switch (Op)
    {
        case Operation::NONE:  break;
        case Operation::ADD:   BackwardAdd();   break;
        case Operation::SUB:   BackwardSub();   break;
        case Operation::MULT:  BackwardMult();  break;
        case Operation::MMULT: BackwardMMult(); break;
        case Operation::POW:   BackwardPow();   break;
        case Operation::TANH:  BackwardTanh();  break;
        case Operation::EXP:   BackwardExp();   break;
    }
}

void Flow::NArrayCore::BackwardAdd()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardAdd(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradients.first;
    if (Operands[1]->Gradient)
        Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArrayCore::BackwardSub()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardSub(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradients.first;
    if (Operands[1]->Gradient)
        Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArrayCore::BackwardMult()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardMult(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradients.first;
    if (Operands[1]->Gradient)
        Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArrayCore::BackwardMMult()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardMMult(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradients.first;
    if (Operands[1]->Gradient)
        Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArrayCore::BackwardPow()
{
    #ifdef FLOW_TORCH_MODE
    auto gradient = TorchBackwardPow(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        2.0f,
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradient;
    #else
    #endif
}

void Flow::NArrayCore::BackwardExp()
{
    #ifdef FLOW_TORCH_MODE
    auto gradient = TorchBackwardExp(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradient;
    #else
    #endif
}

void Flow::NArrayCore::BackwardTanh()
{
    #ifdef FLOW_TORCH_MODE
    vector<float> gradient = TorchBackwardTanh(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    if (Operands[0]->Gradient)
        Operands[0]->Gradient->Data = gradient;
    #else
    #endif
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
    if (first)
        BuildTopo( first, visited, topo );
    if ( current->Operands.size() != 1 )
    {
        NArrayCore* second = current->Operands[1];
        if (second)
            BuildTopo( second, visited, topo );
    }
    topo.push_back(current);
}

namespace Flow
{
    // Defined in "ElementWise.hpp".
    NArrayCore* ElementWise( ... );

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto resultTorch = TorchAdd( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArrayCore( resultTorch.first, resultTorch.second, { arr1, arr2 }, NArrayCore::Operation::ADD );
        #else
        return ElementWise( arr1, arr2, NArrayCore::Operation::ADD );
        #endif
    }

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchSub( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArrayCore( result.first, result.second, { arr1, arr2 }, NArrayCore::Operation::SUB );
        #else
        return ElementWise( arr1, arr2, NArrayCore::Operation::SUB );
        #endif
    }

    NArrayCore* Mult( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchMult( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArrayCore( result.first, result.second, { arr1, arr2 }, NArrayCore::Operation::MULT );
        #else
        #endif
    }

    NArrayCore* Mult( NArrayCore* arr, float literal )
    {
        #ifdef FLOW_TORCH_MODE
        NArrayCore* constant = new NArrayCore( { 1 }, { literal } );
        auto result = TorchMult( { arr->GetShape(), arr->Get() }, { constant->GetShape(), constant->Get() } );
        return new NArrayCore( result.first, result.second, { arr, constant }, NArrayCore::Operation::MULT );
        #else
        #endif
    }

    NArrayCore* MMult( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchMMult( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArrayCore( result.first, result.second, { arr1, arr2 }, NArrayCore::Operation::MMULT );
        #else
        #endif
    }

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        #else
        #endif
        return Mult( arr1, Pow( arr2, -1.0f ) );
    }

    NArrayCore* Pow( NArrayCore* arr, float exponent )
    {
        #ifdef FLOW_TORCH_MODE
        NArrayCore* constant = new NArrayCore( { 1 }, { exponent } );
        auto result = TorchPow( { arr->GetShape(), arr->Get() }, exponent );
        return new NArrayCore( result.first, result.second, { arr, constant }, NArrayCore::Operation::POW );
        #else
        #endif
    }

    NArrayCore* Exp( NArrayCore* arr )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchExp( { arr->GetShape(), arr->Get() } );
        return new NArrayCore( result.first, result.second, { arr }, NArrayCore::Operation::EXP );
        #else
        #endif
    }

    NArrayCore* Tanh( NArrayCore* arr )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchTanh( { arr->GetShape(), arr->Get() } );
        return new NArrayCore( result.first, result.second, { arr }, NArrayCore::Operation::TANH );
        #else
        #endif
    }

    NArrayCore* Neg( NArrayCore* arr )
    {
        vector<float> data( arr->Get().size(), -1.0f );
        return Mult( arr, new NArrayCore( arr->GetShape(), data ) );
    }

    bool Less( NArrayCore* arr1, NArrayCore* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        #else
        #endif
        return false;
    }

    string OperationString( NArrayCore::Operation op )
    {
        switch (op)
        {
            case NArrayCore::Operation::NONE:  return "";
            case NArrayCore::Operation::ADD:   return "+";
            case NArrayCore::Operation::SUB:   return "-";
            case NArrayCore::Operation::MULT:  return "*";
            case NArrayCore::Operation::MMULT: return "mm";
            case NArrayCore::Operation::POW:   return "^";
            case NArrayCore::Operation::TANH:  return "tanh";
            case NArrayCore::Operation::EXP:   return "exp";
        }
        return "";
    }

    NArrayCore* RandomCore( vector<int> shape )
    {
        random_device randomDevice;
        mt19937 generator(randomDevice());
        uniform_real_distribution<float> distribution( -1.0f, 1.0f );
        int size = NArrayCore::SizeFromShape(shape);
        vector<float> data( size, 0.0f );
        for ( int i = 0; i < size; i++ )
            data[i] = distribution(generator);
        return new NArrayCore( shape, data );
    }

    void Log( NArrayCore* arr )
    {
        for ( float value : arr->Get() )
            Log(value);
    }
}