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

float Flow::NArray::Get( vector<int> coordinates )
{
    return Array->Get(coordinates);
}

vector<float> Flow::NArray::Get()
{
    return Array->Get();
}

Flow::NArrayCore* Flow::NArray::GetCore()
{
    return Array;
}

vector<int> Flow::NArray::GetShape()
{
    return Array->GetShape();
}

int Flow::NArray::GetIndex( vector<int> coordinates )
{
    return Array->GetIndex(coordinates);
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

void Flow::NArray::Reshape( vector<int> shape )
{
    Array->Reshape(shape);
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

    NArray Mult( NArray arr1, NArray arr2 )
    {
        return NArray(Mult( arr1.GetCore(), arr2.GetCore() ));
    }

    NArray Mult( NArray arr, float literal )
    {
        return NArray(Mult( arr.GetCore(), literal ));
    }

    NArray MMult( NArray arr1, NArray arr2 )
    {
        return NArray(MMult( arr1.GetCore(), arr2.GetCore() ));
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
        return NArray(Exp( arr.GetCore() ));
    }

    NArray Tanh( NArray arr )
    {
        return NArray(Tanh( arr.GetCore() ));
    }

    NArray Neg( NArray arr )
    {
        return NArray(Neg( arr.GetCore() ));
    }

    bool Less( NArray arr1, NArray arr2 )
    {
        return Less( arr1.GetCore(), arr2.GetCore() );
    }

    NArray Random( vector<int> shape )
    {
        return NArray(RandomCore(shape));
    }

    void Log( NArray arr )
    {
        Log(arr.GetCore());
    }
}