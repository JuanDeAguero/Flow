// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Tanh()
{
    int numPassed = 0;
    int numTests = 1;
    Flow::NArray::Operation op = Flow::NArray::Operation::TANH;

    Test( 1, numPassed,
        Flow::NArray::Create( { 4, 2 }, { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 } ), nullptr, {}, {}, {}, {}, op,
        { 0, 0.0997, 0.1974, 0.2913, 0.3799, 0.4621, 0.5370, 0.6044 },
        { 4, 2 },
        { 1, 0.9901, 0.9610, 0.9151, 0.8556, 0.7864, 0.7116, 0.6347 },
        { 4, 2 }, {}, {} );

    Flow::Print( "Test_Tanh " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}