// Copyright (c) 2023 Juan M. G. de Agüero

#include "NArray.h"

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

vector<int> Flow::NArray::GetShape()
{
    return Array->GetShape();
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
        return NArray(Add( arr1.GetCore(), arr2.GetCore() ));
    }
    
    NArray Sub( NArray arr1, NArray arr2 )
    {
        return NArray(Sub( arr1.GetCore(), arr2.GetCore() ));
    }

    NArray Mul( NArray arr1, NArray arr2 )
    {
        return NArray(Mul( arr1.GetCore(), arr2.GetCore() ));
    }

    NArray Mul( NArray arr, float literal )
    {
        return NArray(Mul( arr.GetCore(), literal ));
    }

    NArray MM( NArray arr1, NArray arr2 )
    {
        return NArray(MM( arr1.GetCore(), arr2.GetCore() ));
    }

    NArray Div( NArray arr1, NArray arr2 )
    {
        return NArray(Div( arr1.GetCore(), arr2.GetCore() ));
    }

    NArray Pow( NArray arr, float exponent )
    {
        return NArray(Pow( arr.GetCore(), exponent ));
    }

    NArray Exp( NArray arr )
    {
        return NArray(Exp(arr.GetCore()));
    }

    NArray Tanh( NArray arr )
    {
        return NArray(Tanh(arr.GetCore()));
    }

    NArray Neg( NArray arr )
    {
        return NArray(Neg(arr.GetCore()));
    }

    bool Less( NArray arr1, NArray arr2 )
    {
        return Less( arr1.GetCore(), arr2.GetCore() );
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