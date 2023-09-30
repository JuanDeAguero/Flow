// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

Flow::NArray::NArray() {}

Flow::NArray::NArray( NArrayCore* arr )
{
    Array = arr;
}

bool Flow::NArray::IsValid()
{
    if (Array) return true;
    else return false;
}

Flow::NArrayCore* Flow::NArray::GetCore()
{
    return Array;
}

float Flow::NArray::Get( vector<int> coordinates )
{
    return Array->Get(coordinates);
}

vector<float> Flow::NArray::Get()
{
    return Array->Get();
}

int Flow::NArray::GetIndex( vector<int> coordinates )
{
    return Array->GetIndex(coordinates);
}

vector<int> Flow::NArray::GetShape()
{
    return Array->GetShape();
}

vector<int> Flow::NArray::GetStride()
{
    return Array->GetStride();
}

Flow::NArray Flow::NArray::GetGradient()
{
    return NArray(Array->GetGradient());
}

void Flow::NArray::Set( vector<int> coordinates, float value )
{
    Array->Set( coordinates, value );
}

void Flow::NArray::Reset( float value )
{
    Array->Reset(value);
}

void Flow::NArray::Backpropagate()
{
    Array->Backpropagate();
}

Flow::NArray Flow::NArray::Copy()
{
    return NArray(Array->Copy());
}

namespace Flow
{
    Flow::NArray Create( vector<int> shape, vector<float> data )
    {
        NArrayCore* arr = new NArrayCore( shape, data );
        return NArray(arr);
    }

    NArray Add( NArray arr1, NArray arr2 )
    {
        Print("Add");
        return NArray( Add( arr1.GetCore(), arr2.GetCore() ) );
    }

    NArray Mul( NArray arr1, NArray arr2 )
    {
        Print("Mul");
        return NArray( Mul( arr1.GetCore(), arr2.GetCore() ) );
    }

    NArray Mul( NArray arr, float literal )
    {
        Print("Mul");
        return NArray( Mul( arr.GetCore(), literal ) );
    }

    NArray MM( NArray arr1, NArray arr2 )
    {
        Print("MM");
        return NArray( MM( arr1.GetCore(), arr2.GetCore() ) );
    }

    NArray Pow( NArray arr, float exponent )
    {
        Print("Pow");
        return NArray( Pow( arr.GetCore(), exponent ) );
    }

    NArray Exp( NArray arr )
    {
        Print("Exp");
        return NArray( Exp(arr.GetCore()) );
    }

    NArray Tanh( NArray arr )
    {
        Print("Tanh");
        return NArray( Tanh(arr.GetCore()) );
    }

    NArray ReLU( NArray arr )
    {
        Print("ReLU");
        return NArray( ReLU(arr.GetCore()) );
    }

    NArray Log( NArray arr )
    {
        Print("Log");
        return NArray( Log(arr.GetCore()) );
    }

    NArray Sum( NArray arr, int dim )
    {
        Print("Sum");
        return NArray( Sum( arr.GetCore(), dim ) );
    }

    NArray Max( NArray arr, int dim )
    {
        Print("Max");
        return NArray( Max( arr.GetCore(), dim ) );
    }

    NArray Transpose( NArray arr, int firstDim, int secondDim )
    {
        Print("Transpose");
        return NArray( Transpose( arr.GetCore(), firstDim, secondDim ) );
    }

    NArray Broadcast( NArray arr, vector<int> shape )
    {
        Print("Broadcast");
        return NArray( Broadcast( arr.GetCore(), shape ) );
    }

    NArray Gather( NArray arr, int dim, NArray index )
    {
        Print("Gather");
        return NArray( Gather( arr.GetCore(), dim, index.GetCore() ) );
    }

    NArray Unsqueeze( NArray arr, int dim )
    {
        Print("Unsqueeze");
        return NArray( Unsqueeze( arr.GetCore(), dim ) );
    }

    NArray Index( NArray arr, int dim, NArray index )
    {
        Print("Index");
        return NArray( Index( arr.GetCore(), dim, index.GetCore() ) );
    }

    NArray Neg( NArray arr )
    {
        Print("Neg");
        return NArray( Neg(arr.GetCore()) );
    }

    NArray Sub( NArray arr1, NArray arr2 )
    {
        Print("Sub");
        return NArray( Sub( arr1.GetCore(), arr2.GetCore() ) );
    }

    NArray Div( NArray arr1, NArray arr2 )
    {
        Print("Div");
        return NArray( Div( arr1.GetCore(), arr2.GetCore() ) );
    }

    NArray Mean( NArray arr )
    {
        Print("Mean");
        return NArray( Mean(arr.GetCore()) );
    }

    NArray Softmax( NArray arr )
    {
        Print("Softmax");
        return Softmax(arr.GetCore());
    }

    NArray CrossEntropy( NArray arr1, NArray arr2 )
    {
        Print("CrossEntropy");
        return CrossEntropy( arr1.GetCore(), arr2.GetCore() );
    }

    NArray Random( vector<int> shape )
    {
        return NArray(RandomCore(shape));
    }

    void Print( NArray arr )
    {
        Print(arr.GetCore());
    }
}