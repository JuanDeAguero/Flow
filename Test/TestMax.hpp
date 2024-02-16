// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Max()
{
    int numPassed = 0;
    int numTests = 1;
    Flow::NArray::Operation op = Flow::NArray::Operation::MAX;

    Test( 1, numPassed,
        Flow::NArray::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), nullptr,
        { 0 }, {}, {}, {}, op,
        { 6, 7, 8 },
        { 1, 3 },
        { 0, 0, 0, 0, 0, 0, 1, 1, 1 },
        { 3, 3 }, {}, {} );

    Flow::Print( "Test_Max " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}