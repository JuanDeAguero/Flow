// Copyright (c) 2023 Juan M. G. de Agüero

#include <string>

#pragma once

namespace Flow
{
    using namespace std;

    void Print( string message );

    void Print( float value );
    
    void Print( float value, int precision );

    void Print( vector<float> vec );
}