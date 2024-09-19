// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::OneHot( vector<int> integers, int num )
{
    vector<float> data( integers.size() * num );
    NARRAY arr = Create( { (int)integers.size(), num }, data );
    for ( int i = 0; i < integers.size(); i++ )
    {
        for ( int j = 0; j < num; j++ )
        {
            float value = 0.0f;
            if ( integers[i] == j ) value = 1.0f;
            arr->Set( { i, j }, value );
        }
    }
    return arr;
}