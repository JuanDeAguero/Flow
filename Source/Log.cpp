// Copyright (c) 2023 Juan M. G. de Agüero

#include <iomanip>
#include <iostream>

#include "Log.h"
#include "NArray.h"

using namespace std;

void Flow::Log( string message )
{
    cout << message << endl;
}

void Flow::Log( float value )
{
    Log( value, 4 );
}

void Flow::Log( float value, int precision )
{
    ios_base::fmtflags originalFlags = cout.flags();
    cout << fixed << setprecision(precision) << value << endl;
    cout.flags(originalFlags);
}

void Flow::Log( vector<float> vec )
{
    for ( float value : vec )
        Log(value);
}


void Flow::Log( NArray* arr )
{
    for ( float value : arr->Get() )
        Log(value, 20);
}