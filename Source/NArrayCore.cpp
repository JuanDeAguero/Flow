// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "NArray.h"
#include "Print.h"

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

vector<int> Flow::NArrayCore::GetShape()
{
    return Shape;
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

void Flow::NArrayCore::Backward()
{
    if ( Operands.size() == 0 )
        return;
    switch (Op)
    {
        case Operation::NONE: break;

        case Operation::ADD:       BackwardAdd();       break;
        case Operation::MUL:       BackwardMul();       break;
        case Operation::MM:        BackwardMM();        break;
        case Operation::POW:       BackwardPow();       break;
        case Operation::EXP:       BackwardExp();       break;
        case Operation::TANH:      BackwardTanh();      break;
        case Operation::RESHAPE:   BackwardReshape();   break;
        case Operation::BROADCAST: BackwardBroadcast(); break;
    }
}

void Flow::NArrayCore::BackwardAdd()
{
    if ( Operands.size() != 2 )
    {
        Print("[Error] Invalid number of operands in BackwardAdd.");
        return;
    }
    NArrayCore* operand1 = Operands[0];
    NArrayCore* operand2 = Operands[1];
    if ( Gradient->Data.size() != operand1->Gradient->Data.size() || 
        Gradient->Data.size() != operand2->Gradient->Data.size() )
    {
        Print("[Error] Invalid operand gradient in BackwardAdd.");
        return;
    }
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        operand1->Gradient->Data[i] += Gradient->Data[i];
        operand2->Gradient->Data[i] += Gradient->Data[i];
    }
}

void Flow::NArrayCore::BackwardMul()
{

}

void Flow::NArrayCore::BackwardMM()
{

}

void Flow::NArrayCore::BackwardPow()
{

}

void Flow::NArrayCore::BackwardExp()
{

}

void Flow::NArrayCore::BackwardTanh()
{

}

void Flow::NArrayCore::BackwardReshape()
{

}

void Flow::NArrayCore::BackwardBroadcast()
{
    if ( Operands.size() != 1 )
    {
        Print("[Error] Invalid number of operands in BackwardBroadcast.");
        return;
    }
    NArrayCore* operand = Operands[0];
    vector<float> operandGradient( operand->Data.size(), 0.0f );
    vector<int> operandShape = operand->Shape;
    vector<int> shape = this->Shape;
    vector<int> position( shape.size(), 0 );
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        int index = 0;
        for ( int j = 0; j < operandShape.size(); j++ )
        {
            int coord = position[ shape.size() - operandShape.size() + j ];
            if ( operandShape[j] == 1 )
                coord = 0;
            index += coord * operand->Stride[j];
        }
        operandGradient[index] += Gradient->Data[i];
        for ( int j = shape.size() - 1; j >= 0; j-- )
        {
            position[j]++;
            if ( position[j] < shape[j] ) break;
            else position[j] = 0;
        }
    }
    for ( int i = 0; i < operand->Gradient->Data.size(); i++ )
        operand->Gradient->Data[i] += operandGradient[i];
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
    vector<int> GetShapeForBroadcast( NArrayCore* arr1, NArrayCore* arr2 )
    {
        vector<int> shape1 = arr1->GetShape();
        vector<int> shape2 = arr2->GetShape();
        int maxDims = max( shape1.size(), shape2.size() );
        while ( shape1.size() < maxDims ) shape1.insert( shape1.begin(), 1 );
        while ( shape2.size() < maxDims ) shape2.insert( shape2.begin(), 1 );
        vector<int> shape(maxDims);
        for ( int i = 0; i < maxDims; i++ )
        {
            if ( shape1[i] == shape2[i] ) shape[i] = shape1[i];
            else if ( shape1[i] == 1 ) shape[i] = shape2[i];
            else if ( shape2[i] == 1 ) shape[i] = shape1[i];
            else
            {
                Print("[Error] The arrays are not compatible for broadcast.");
                return {};
            }
        }
        return shape;
    }

    void ElementWise( vector<int> index, NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        if ( index.size() == arr1->GetShape().size() )
        {
            switch (op)
            {
                case NArrayCore::Operation::ADD:
                    result->Set( index, arr1->Get(index) + arr2->Get(index) );
                    break;
                case NArrayCore::Operation::MUL:
                    result->Set( index, arr1->Get(index) * arr2->Get(index) );
                    break;
            }
            return;
        }
        for ( int i = 0; i < arr1->GetShape()[index.size()]; i++ )
        {
            std::vector<int> newIndex = index;
            newIndex.push_back(i);
            ElementWise( newIndex, arr1, arr2, result, op );
        }
    }

    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 )
    {
        auto shape = GetShapeForBroadcast( arr1, arr2 );
        Flow::NArrayCore* arr1B = Flow::Broadcast( arr1, shape );
        Flow::NArrayCore* arr2B = Flow::Broadcast( arr2, shape );
        auto op = NArrayCore::Operation::ADD;
        NArrayCore* result = new NArrayCore( arr1B->GetShape(), arr1B->Get(), { arr1B, arr2B }, op );
        ElementWise( {}, arr1B, arr2B, result, op );
        return result;
    }

    NArrayCore* Sub( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return Add( arr1, Neg(arr2) );
    }

    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 )
    {
        auto shape = GetShapeForBroadcast( arr1, arr2 );
        Flow::NArrayCore* arr1B = Flow::Broadcast( arr1, shape );
        Flow::NArrayCore* arr2B = Flow::Broadcast( arr2, shape );
        auto op = NArrayCore::Operation::MUL;
        NArrayCore* result = new NArrayCore( arr1B->GetShape(), arr1B->Get(), { arr1B, arr2B }, op );
        ElementWise( {}, arr1B, arr2B, result, op );
        return result;
    }

    NArrayCore* Mul( NArrayCore* arr, float literal )
    {
        NArrayCore* arrLiteral = new NArrayCore( { 1 }, { literal } );
        return Mul( arr, arrLiteral );
    }

    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 )
    {
        
    }

    NArrayCore* Div( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return Mul( arr1, Pow( arr2, -1.0f ) );
    }

    NArrayCore* Pow( NArrayCore* arr, float exponent )
    {
        
    }

    NArrayCore* Exp( NArrayCore* arr )
    {
        
    }

    NArrayCore* Tanh( NArrayCore* arr )
    {
        
    }

    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape )
    {
        return new NArrayCore( shape, arr->Get(), { arr }, NArrayCore::Operation::RESHAPE );
    }

    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {

    }

    NArrayCore* Flow::Broadcast( NArrayCore* arr, vector<int> shape )
    {
        if ( shape.size() < arr->GetShape().size() )
        {
            Print("[Error] Incompatible shape for broadcast.");
            return nullptr;
        }
        for (int i = 1; i <= arr->GetShape().size(); i++)
        {
            if ( shape[ shape.size() - i ] != arr->GetShape()[ arr->GetShape().size() - i ] &&
                arr->GetShape()[ arr->GetShape().size() - i ] != 1 &&
                shape[ shape.size() - i ] != 1 )
            {
                Print("[Error] Incompatible shape for broadcast.");
                return nullptr;
            }
        }

        vector<float> data( SizeFromShape(shape), 0.0f );
        vector<int> position( shape.size(), 0 );
        for ( int i = 0; i < SizeFromShape(shape); i++ )
        {
            vector<int> originalCoords;
            for ( int j = 0; j < arr->GetShape().size(); j++ )
            {
                int coord = position[ shape.size() - arr->GetShape().size() + j ];
                if ( arr->GetShape()[j] == 1 )
                    coord = 0;
                originalCoords.push_back(coord);
            }
            data[i] = arr->Get(originalCoords);
            for (int j = shape.size() - 1; j >= 0; j-- )
            {
                position[j]++;
                if (position[j] < shape[j]) break;
                else position[j] = 0;
            }
        }

        return new NArrayCore( shape, data, { arr }, NArrayCore::Operation::BROADCAST );
    }

    NArrayCore* Neg( NArrayCore* arr )
    {
        return Mul( arr, -1.0f );
    }

    bool Less( NArrayCore* arr1, NArrayCore* arr2 )
    {
        return false;
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
}