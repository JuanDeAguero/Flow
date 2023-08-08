// Copyright (c) 2023 Juan M. G. de Agüero

#include <string>

#pragma once

namespace Flow
{
    using namespace std;

    void Log( string message );

    void Log( float value );
    
    void Log( float value, int precision );

    void Log( vector<float> vec );

    void Log( class NArray* arr );
}