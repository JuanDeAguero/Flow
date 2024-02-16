// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Mul()
{
    int numPassed = 0;
    int numTests = 2;
    Flow::NArray::Operation op = Flow::NArray::Operation::MUL;

    Test( 1, numPassed,
        Flow::NArray::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ),
        Flow::NArray::Create( { 3 }, { 1, 10, 100 } ), {}, {}, {}, {}, op,
        { 0, 10, 200, 3, 40, 500, 6, 70, 800 },
        { 3, 3 },
        { 1, 10, 100, 1, 10, 100, 1, 10, 100 },
        { 3, 3 },
        { 9, 12, 15 },
        { 3 } );

    Test( 2, numPassed,
        Flow::NArray::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ),
        Flow::NArray::Create( { 1 }, { 2 } ), {}, {}, {}, {}, op,
        { 0, 2, 4, 6, 8, 10, 12, 14, 16 },
        { 3, 3 },
        { 2, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 3, 3 },
        { 36 },
        { 1 } );

    Flow::Print( "Test_Mul " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}