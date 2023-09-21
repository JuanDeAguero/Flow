// Copyright (c) Juan M. G. de Agüero 2023

#include "Vector.h"

using namespace std;

namespace Flow
{
    bool Equals( vector<float> vec1, vector<float> vec2, float tolerance )
    {
        if ( vec1.size() != vec2.size() )
            return false;
        for ( int i = 0; i < vec1.size(); i++ )
        {
            if ( !( fabs(vec1[i] - vec2[i]) < tolerance ) )
                return false;
        }
        return true;
    }
}