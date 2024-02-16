// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"
#include "Test.hpp"

static bool Test_MM()
{
    int numPassed = 0;
    int numTests = 3;
    Flow::NArray::Operation op = Flow::NArray::Operation::MM;

    Test( 1, numPassed,
        Flow::NArray::Create( { 2, 2 }, { 1, 2, 3, 4 } ),
        Flow::NArray::Create( { 2, 2 }, { 5, 6, 7, 8 } ), {}, {}, {}, {}, op,
        { 19, 22, 43, 50 },
        { 2, 2 },
        { 11, 15, 11, 15 },
        { 2, 2 },
        { 4, 4, 6, 6 },
        { 2, 2 } );

    Test( 2, numPassed,
        Flow::NArray::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } ),
        Flow::NArray::Create( { 4, 2 }, { 5, 6, 5, 6, 1, 3, 7, 8 } ), {}, {}, {}, {}, op,
        { 131, 168, 68, 89 },
        { 2, 2 },
        { 11, 11, 4, 15, 11, 11, 4, 15 },
        { 2, 4 },
        { 8, 8, 11, 11, 13, 13, 13, 13 },
        { 4, 2 } );

    Test( 3, numPassed,
        Flow::NArray::Create( { 2, 3 }, { 1, 2, 3, 4, 5, 6 } ),
        Flow::NArray::Create( { 3, 2 }, { 1, 2, 3, 4, 5, 6 } ), {}, {}, {}, {}, op,
        { 22, 28, 49, 64 },
        { 2, 2 },
        { 3, 7, 11, 3, 7, 11 },
        { 2, 3 },
        { 5, 5, 7, 7, 9, 9 },
        { 3, 2 } );

    Flow::Print( "Test_MM " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}