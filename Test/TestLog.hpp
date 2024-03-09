// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Log()
{
    int numPassed = 0;
    int numTests = 1;
    Flow::NArray::Operation op = Flow::NArray::Operation::LOG;

    Test( 1, numPassed,
        Flow::Create( { 4, 2 }, { 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 } ), nullptr, {}, {}, {}, {}, op,
        { 1.6094, -2.3026, -1.6094, -1.2040, -0.9163, -0.6931, -0.5108, -0.3567 },
        { 4, 2 },
        { 0.2, 10, 5, 3.3333, 2.5, 2, 1.6667, 1.4286 },
        { 4, 2 }, {}, {} );

    Flow::Print( "Test_Log " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}