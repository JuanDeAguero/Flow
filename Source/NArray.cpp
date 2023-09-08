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

Flow::NArray::NArray( vector<int> shape, vector<float> data )
{
    Data = data;
    Shape = shape;
    Constant = false;
    vector<float> gradientData( SizeFromShape(shape), 0.0f );
    Gradient = new NArray( shape, gradientData, true );
    Op = Operation::NONE;
}

Flow::NArray::NArray( vector<int> shape, vector<float> data, bool constant )
{
    Data = data;
    Shape = shape;
    Constant = constant;
    if (!Constant)
    {
        vector<float> gradientData( SizeFromShape(shape), 0.0f );
        Gradient = new NArray( shape, gradientData, true );
    }
    Op = Operation::NONE;
}

Flow::NArray::NArray( vector<int> shape, vector<float> data, vector<NArray*> operands, Operation op )
{
    Data = data;
    Shape = shape;
    Constant = false;
    vector<float> gradientData( SizeFromShape(shape), 0.0f );
    Gradient = new NArray( shape, gradientData, true );
    Operands = operands;
    Op = op;
}

float Flow::NArray::Get( vector<int> coordinates )
{
    int index = GetIndex(coordinates);
    if ( index >= 0 && index < Data.size() )
        return Data[index];
    return 0.0f;
}

vector<float> Flow::NArray::Get()
{
    return Data;
}

vector<int> Flow::NArray::GetShape()
{
    return Shape;
}

int Flow::NArray::GetIndex( vector<int> coordinates )
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

Flow::NArray* Flow::NArray::GetGradient()
{
    return Gradient;
}

void Flow::NArray::Set( vector<int> coordinates, float value )
{
    int index = GetIndex(coordinates);
    if ( index >= 0 && index < Data.size() )
        Data[index] = value;
}

void Flow::NArray::Reset( float value )
{
    for ( int i = 0; i < Data.size(); i++ )
        Data[i] = value;
}

void Flow::NArray::Reshape( vector<int> shape )
{
    Shape = shape;
}

void Flow::NArray::Backpropagate()
{
    Gradient->Reset(1.0f);
    TopologicalSort();
    for ( Flow::NArray* arr : TopologicalSort() )
        arr->Backward();
}

string Flow::NArray::Trace()
{
    set<NArray*> nodes;
    set< pair< NArray*, NArray* > > edges;
    BuildGraph( this, nodes, edges );
    ostringstream trace;
    for ( NArray* node : nodes )
    {
        //trace << ;
        //trace << ;
        trace << OperationString(node->Op) << endl;
    }
    for ( auto& edge : edges )
        trace << edge.first << " -> " << edge.second << endl;
    return trace.str();
}

int Flow::NArray::SizeFromShape( vector<int> shape )
{
    int size = shape[0];
    for ( int i = 1; i < shape.size(); i++ )
        size *= shape[i];
    return size;
}

Flow::NArray* Flow::NArray::Copy()
{
    NArray* copy = new NArray( Shape, Data );
    copy->Constant = true;
    return copy;
}

void Flow::NArray::Backward()
{
    if ( Operands.size() == 0 || Constant )
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

void Flow::NArray::BackwardAdd()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardAdd(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradients.first;
    Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArray::BackwardSub()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardSub(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradients.first;
    Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArray::BackwardMult()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardMult(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradients.first;
    Operands[1]->Gradient->Data = gradients.second;
    #else
    if ( Operands[1]->Constant )
        Operands[0]->Gradient->Data = Gradient->Data;
    else
    {

    }
    #endif
}

void Flow::NArray::BackwardMMult()
{
    #ifdef FLOW_TORCH_MODE
    auto gradients = TorchBackwardMMult(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Operands[1]->GetShape(), Operands[1]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradients.first;
    Operands[1]->Gradient->Data = gradients.second;
    #else
    #endif
}

void Flow::NArray::BackwardPow()
{
    #ifdef FLOW_TORCH_MODE
    auto gradient = TorchBackwardPow(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        2.0f,
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradient;
    #else
    #endif
}

void Flow::NArray::BackwardExp()
{
    #ifdef FLOW_TORCH_MODE
    auto gradient = TorchBackwardExp(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradient;
    #else
    #endif
}

void Flow::NArray::BackwardTanh()
{
    #ifdef FLOW_TORCH_MODE
    vector<float> gradient = TorchBackwardTanh(
        { Operands[0]->GetShape(), Operands[0]->Get() },
        { Gradient->GetShape(), Gradient->Get() });
    Operands[0]->Gradient->Data = gradient;
    #else
    #endif
}

vector<Flow::NArray*> Flow::NArray::TopologicalSort()
{
    unordered_set<NArray*> visited;
    vector<NArray*> topo;
    BuildTopo( this, visited, topo );
    reverse( topo.begin(), topo.end() );
    return topo;
}

void Flow::NArray::BuildTopo( NArray* current, unordered_set<NArray*>& visited, vector<NArray*>& topo )
{
    if ( visited.find(current) != visited.end() || current->Operands.size() == 0 )
        return;
    visited.insert(current);
    NArray* first = current->Operands[0];
    if ( first && !first->Constant )
        BuildTopo( first, visited, topo );
    if ( current->Operands.size() != 1 )
    {
        NArray* second = current->Operands[1];
        if ( second && !second->Constant )
            BuildTopo( second, visited, topo );
    }
    topo.push_back(current);
}

void Flow::NArray::BuildGraph( NArray* current, set<NArray*>& nodes, set< pair< NArray*, NArray* > >& edges )
{
    if ( nodes.find(current) != nodes.end() || current->Operands.size() == 0 )
        return;
    nodes.insert(current);
    NArray* first = current->Operands[0];
    if ( first && !first->Constant )
    {
        edges.insert({ first, current });
        BuildGraph( first, nodes, edges );
    }
    if ( current->Operands.size() != 1 )
    {
        NArray* second = current->Operands[1];
        if ( second  && !second->Constant )
        {
            edges.insert({ second, current });
            BuildGraph( second, nodes, edges );
        }
    }
}

namespace Flow
{
    // Defined in "ElementWise.hpp".
    NArray* ElementWise( ... );

    NArray* Add( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto resultTorch = TorchAdd( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArray( resultTorch.first, resultTorch.second, { arr1, arr2 }, NArray::Operation::ADD );
        #else
        return ElementWise( arr1, arr2, NArray::Operation::ADD );
        #endif
    }

    NArray* Sub( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchSub( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArray( result.first, result.second, { arr1, arr2 }, NArray::Operation::SUB );
        #else
        return ElementWise( arr1, arr2, NArray::Operation::SUB );
        #endif
    }

    NArray* Mult( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchMult( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArray( result.first, result.second, { arr1, arr2 }, NArray::Operation::MULT );
        #else
        #endif
    }

    NArray* Mult( NArray* arr, float literal )
    {
        #ifdef FLOW_TORCH_MODE
        NArray* constant = new NArray( { 1 }, { literal }, true );
        auto result = TorchMult( { arr->GetShape(), arr->Get() }, { constant->GetShape(), constant->Get() } );
        return new NArray( result.first, result.second, { arr, constant }, NArray::Operation::MULT );
        #else
        #endif
    }

    NArray* MMult( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchMMult( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArray( result.first, result.second, { arr1, arr2 }, NArray::Operation::MMULT );
        #else
        #endif
    }

    NArray* Div( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        #else
        #endif
        return Mult( arr1, Pow( arr2, -1.0f ) );
    }

    NArray* Pow( NArray* arr, float exponent )
    {
        #ifdef FLOW_TORCH_MODE
        NArray* constant = new NArray( { 1 }, { exponent }, true );
        auto result = TorchPow( { arr->GetShape(), arr->Get() }, exponent );
        return new NArray( result.first, result.second, { arr, constant }, NArray::Operation::POW );
        #else
        #endif
    }

    NArray* Exp( NArray* arr )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchExp( { arr->GetShape(), arr->Get() } );
        return new NArray( result.first, result.second, { arr }, NArray::Operation::EXP );
        #else
        #endif
    }

    NArray* Tanh( NArray* arr )
    {
        #ifdef FLOW_TORCH_MODE
        auto result = TorchTanh( { arr->GetShape(), arr->Get() } );
        return new NArray( result.first, result.second, { arr }, NArray::Operation::TANH );
        #else
        #endif
    }

    NArray* Neg( NArray* arr )
    {
        vector<float> data( arr->Get().size(), -1.0f );
        return Mult( arr, new NArray( arr->GetShape(), data, true ) );
    }

    bool Less( NArray* arr1, NArray* arr2 )
    {
        #ifdef FLOW_TORCH_MODE
        #else
        #endif
        return false;
    }

    string OperationString( NArray::Operation op )
    {
        switch (op)
        {
            case NArray::Operation::NONE:  return "";
            case NArray::Operation::ADD:   return "+";
            case NArray::Operation::SUB:   return "-";
            case NArray::Operation::MULT:  return "*";
            case NArray::Operation::MMULT: return "mm";
            case NArray::Operation::POW:   return "^";
            case NArray::Operation::TANH:  return "tanh";
            case NArray::Operation::EXP:   return "exp";
        }
        return "";
    }

    NArray* Random( vector<int> shape )
    {
        random_device randomDevice;
        mt19937 generator(randomDevice());
        uniform_real_distribution<float> distribution( -1.0f, 1.0f );
        int size = NArray::SizeFromShape(shape);
        vector<float> data( size, 0.0f );
        for ( int i = 0; i < size; i++ )
            data[i] = distribution(generator);
        return new NArray( shape, data );
    }
}