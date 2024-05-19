// Copyright (c) Juan M. G. de Agüero 2023

#include <cmath>

#include "Vector.h"

using namespace std;

bool Flow::Equals( vector<float> vec1, vector<float> vec2, float tolerance )
{
    if ( vec1.size() != vec2.size() ) return false;
    for ( int i = 0; i < vec1.size(); i++ )
    {
        if ( !( fabs( vec1[i] - vec2[i] ) < tolerance ) )
            return false;
    }
    return true;
}

vector<int> Flow::ToInt( vector<float> vec )
{
    vector<int> vecInt;
    for ( float value : vec ) vecInt.push_back((int)value);
    return vecInt;
}

vector<int64_t> Flow::ToInt64( vector<int> vec )
{
    vector<int64_t> vecInt64;
    for ( int value : vec ) vecInt64.push_back((int64_t)value);
    return vecInt64;
}