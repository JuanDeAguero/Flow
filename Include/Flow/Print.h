// Copyright (c) 2023-2024 Juan M. G. de Agüero

#pragma once

#include <string>
#include <vector>

namespace Flow
{
    using namespace std;

    void Print( string message );

    void Print( float value );
    
    void Print( float value, int precision );

    void Print( vector<float> vec );

    void Print( vector<int> vec );
}