// Copyright (c) Juan M. G. de Agüero 2023

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "NArray.h"
#include "Log.h"

#include "Torch.hpp"

using namespace std;

Flow::NArray::NArray( float value )
{
    Data = { value };
    Shape = { 1 };
    Constant = false;
    Gradient = { 0.0f };
    Op = Operation::NONE;
}

Flow::NArray::NArray( vector<int> shape, vector<float> data )
{
    Data = data;
    Shape = shape;
    Constant = false;
    Gradient.resize( Data.size(), 0.0f );
    Op = Operation::NONE;
}

Flow::NArray::NArray( vector<int> shape, vector<float> data, bool constant )
{
    Data = data;
    Shape = shape;
    Constant = constant;
    Gradient.resize( Data.size(), 0.0f );
    Op = Operation::NONE;
}

Flow::NArray::NArray( vector<int> shape, vector<float> data, vector<NArray*> operands, Operation op )
{
    Data = data;
    Shape = shape;
    Constant = false;
    Gradient.resize( Data.size(), 0.0f );
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

vector<float> Flow::NArray::GetGradient()
{
    return Gradient;
}

void Flow::NArray::Set( vector<int> coordinates, float value )
{
    int index = GetIndex(coordinates);
    if ( index >= 0 && index < Data.size() )
        Data[index] = value;
}

void Flow::NArray::Reshape( vector<int> shape )
{
    Shape = shape;
}

void Flow::NArray::ResetGradient()
{
    for ( int i = 0; i < Gradient.size(); i++ )
        Gradient[i] = 0.0f;
}

void Flow::NArray::Backpropagate()
{
    vector<float> gradient;
    gradient.resize( Data.size(), 1.0f );
    Gradient = gradient;
    for ( Flow::NArray* arr : TopologicalSort() )
        arr->Backward();
}

void Flow::NArray::Backward()
{
    if ( Operands.size() == 0 )
        return;
    switch (Op)
    {
        case Operation::NONE:                     break;
        case Operation::ADD:   { BackwardAdd();   break; }
        case Operation::MULT:  { BackwardMult();  break; }
        case Operation::MMULT: { BackwardMMult(); break; }
        case Operation::POW:   { BackwardPow();   break; }
        case Operation::TANH:  { BackwardTanh();  break; }
        case Operation::EXP:   { BackwardExp();   break; }
    }
}

void Flow::NArray::BackwardAdd()
{

}

void Flow::NArray::BackwardMult()
{

}

void Flow::NArray::BackwardMMult()
{

}

void Flow::NArray::BackwardPow()
{

}

void Flow::NArray::BackwardTanh()
{

}

void Flow::NArray::BackwardExp()
{
    
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
    if ( visited.find(current) != visited.end() )
        return;
    visited.insert(current);
    NArray* first = current->Operands[0];
    NArray* second = current->Operands[1];
    if ( first && !first->Constant )
        BuildTopo( first, visited, topo );
    if ( second && !second->Constant )
        BuildTopo( second, visited, topo );
    topo.push_back(current);
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

void Flow::NArray::BuildGraph( NArray* current, set<NArray*>& nodes, set< pair< NArray*, NArray* > >& edges )
{
    if ( nodes.find(current) != nodes.end() )
        return;
    nodes.insert(current);
    NArray* first = current->Operands[0];
    NArray* second = current->Operands[1];
    if ( first && !first->Constant )
    {
        edges.insert({ first, current });
        BuildGraph( first, nodes, edges );
    }
    if ( second  && !second->Constant )
    {
        edges.insert({ second, current });
        BuildGraph( second, nodes, edges );
    }
}

namespace Flow
{
    NArray* Add( NArray* arr1, NArray* arr2 )
    {
        if ( arr1->GetShape().size() > 2 ||  arr2->GetShape().size() > 2 )
        {
            Log("[Error] Only 1D and 2D arrays are supported for addition.");
            return nullptr;
        }

        // Create a copy of the two arrays.
        // They might need to be reshaped and we don't want to modify the input arrays.
        NArray* arr1Copy = new NArray( arr1->GetShape(), arr1->Get(), true );
        NArray* arr2Copy = new NArray( arr2->GetShape(), arr2->Get(), true );

        // Add 1s if needed.
        if ( arr1->GetShape().size() == 2 && arr2->GetShape().size() == 1 )
            arr2Copy->Reshape({ 1, arr2->GetShape()[1] });
        else if ( arr1->GetShape().size() == 1 && arr2->GetShape().size() == 2 )
            arr1Copy->Reshape({ 1, arr1->GetShape()[1] });

        // Check if shapes are compatible.
        for ( int i = 0; i < arr1Copy->GetShape().size(); i++ )
        {
            if ( arr1Copy->GetShape()[i] != arr2Copy->GetShape()[i]
                && arr1Copy->GetShape()[i] != 1 && arr2Copy->GetShape()[i] != 1 )
            {
                Log("[Error] Array shapes are incompatible for addition.");
                return nullptr;
            }
        }

        // Create the result array.
        vector<int> resultShape;
        for ( int i = 0; i < arr1Copy->GetShape().size(); i++ )
        {
            if ( arr1Copy->GetShape()[i] != 1 )
                resultShape.push_back(arr1Copy->GetShape()[i]);
            else resultShape.push_back(arr2Copy->GetShape()[i]);
        }
        int resultSize = resultShape[0];
        for ( int i = 1; i < resultShape.size(); i++ )
            resultSize *= resultShape[i];
        vector<float> resultData( resultSize, 0.0f );
        NArray* result = new NArray( resultShape, resultData, { arr1, arr2 }, NArray::Operation::ADD );

        // The two arrays have compatible shapes so we can add them.
        vector<int> coords1;
        vector<int> coords2;
        for ( int i = 0; i < resultShape[0]; i++ )
        {
            if ( arr1Copy->GetShape().size() == 1 )
                result->Set( { i }, arr1Copy->Get({ i }) + arr2Copy->Get({ i }) );
            else
            {
                for ( int j = 0; j < resultShape[1]; j++ )
                {
                    coords1 = { i, j };
                    coords2 = { i, j };
                    if ( arr1Copy->GetShape()[0] == 1 ) coords1[0] = 0;
                    if ( arr1Copy->GetShape()[1] == 1 ) coords1[1] = 0;
                    if ( arr2Copy->GetShape()[0] == 1 ) coords2[0] = 0;
                    if ( arr2Copy->GetShape()[1] == 1 ) coords2[1] = 0;
                    result->Set( { i, j }, arr1Copy->Get(coords1) + arr2Copy->Get(coords2) );
                }
            }
        }
        return result;
    }

    NArray* Sub( NArray* arr1, NArray* arr2 )
    {
        return Add( arr1, Neg(arr2) );
    }

    NArray* Sub( NArray* arr, float literal )
    {
        NArray* constant = new NArray( { 1 }, { -1.0f * literal }, true );
        return Add( arr, constant );
    }

    NArray* Neg( NArray* arr )
    {
        vector<float> data( arr->Get().size(), -1.0f );
        return Mult( arr, new NArray( arr->GetShape(), data, true ) );
    }

    NArray* Mult( NArray* arr1, NArray* arr2 )
    {
        auto result = TorchMult( { arr1->GetShape(), arr1->Get() }, { arr2->GetShape(), arr2->Get() } );
        return new NArray( result.first, result.second, { arr1, arr2 }, NArray::Operation::MULT );
    }

    NArray* Mult( NArray* arr, float literal )
    {
        return new NArray( {}, {}, {}, NArray::Operation::MULT );
    }

    NArray* MMult( NArray* arr1, NArray* arr2 )
    {
        return new NArray( {}, {}, {}, NArray::Operation::MMULT );
    }

    NArray* Div( NArray* arr1, NArray* arr2 )
    {
        return Mult( arr1, Pow( arr2, -1.0f ) );
    }

    NArray* Pow( NArray* arr, float exponent )
    {
        return new NArray( {}, {}, {}, NArray::Operation::POW );
    }

    bool Less( NArray* arr1, NArray* arr2 )
    {
        return false;
    }

    NArray* Tanh( NArray* arr )
    {
        return new NArray( {}, {}, {}, NArray::Operation::TANH );
    }

    NArray* Exp( NArray* arr )
    {
        return new NArray( {}, {}, {}, NArray::Operation::EXP );
    }

    string OperationString( NArray::Operation op )
    {
        switch (op)
        {
            case NArray::Operation::NONE:  return "";
            case NArray::Operation::ADD:   return "+";
            case NArray::Operation::MULT:  return "*";
            case NArray::Operation::MMULT: return "mm";
            case NArray::Operation::POW:   return "^";
            case NArray::Operation::TANH:  return "tanh";
            case NArray::Operation::EXP:   return "exp";
        }
        return "";
    }
}