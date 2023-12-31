// Copyright (c) 2023 Juan M. G. de Agüero

#include <iomanip>
#include <iostream>

#include "Flow/Print.h"

using namespace std;

void Flow::Print( string message )
{
    cout << message << endl;
}

void Flow::Print( float value )
{
    Print( value, 4 );
}

void Flow::Print( float value, int precision )
{
    ios_base::fmtflags flags = cout.flags();
    cout << fixed << setprecision(precision) << value << endl;
    cout.flags(flags);
}

void Flow::Print( vector<float> vec )
{
    for ( float value : vec ) Print(value);
}

void Flow::Print( vector<int> vec )
{
    for ( int value : vec ) Print(value);
}