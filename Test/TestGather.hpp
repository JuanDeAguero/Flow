// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Gather()
{
    int numPassed = 0;
    int numTests = 7;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::GATHER;

    Test( 1, numPassed,
        Flow::Create( { 3, 2 }, { 0, 1, 10, 11, 20, 21 } ),
        Flow::Create( { 2, 2 }, { 2, 1, 1, 0 } ),
        { 0 }, {}, {}, {}, op,
        { 20, 11, 10, 1 },
        { 2, 2 },
        { 0, 1, 1, 1, 1, 0 },
        { 3, 2 },
        { 0, 0, 0, 0 },
        { 2, 2 } );

    Test( 2, numPassed,
        Flow::Create( { 3, 2 }, { 0, 1, 10, 11, 20, 21 } ),
        Flow::Create( { 3, 1 }, { 1, 0, 1 } ),
        { 1 }, {}, {}, {}, op,
        { 1, 10, 21 },
        { 3, 1 },
        { 0, 1, 1, 0, 0, 1 },
        { 3, 2 },
        { 0, 0, 0 },
        { 3, 1 } );

    Test( 3, numPassed,
        Flow::Create( { 4 }, { 0, 10, 20, 30 } ),
        Flow::Create( { 3 }, { 3, 0, 2 } ),
        { 0 }, {}, {}, {}, op,
        { 30, 0, 20 },
        { 3 },
        { 1, 0, 1, 1 },
        { 4 },
        { 0, 0, 0 },
        { 3 } );

    Test( 4, numPassed,
        Flow::Create( { 3, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22 } ),
        Flow::Create( { 2, 2 }, { 2, 1, 0, 2 } ),
        { 1 }, {}, {}, {}, op,
        { 2, 1, 10, 12 },
        { 2, 2 },
        { 0, 1, 1, 1, 0, 1, 0, 0, 0 },
        { 3, 3 },
        { 0, 0, 0, 0 },
        { 2, 2 } );

    Test( 5, numPassed,
        Flow::Create( { 2, 2, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 } ),
        Flow::Create( { 2, 2, 2 }, { 2, 1, 1, 0, 0, 2, 2, 1 } ),
        { 2 }, {}, {}, {}, op,
        { 2, 1, 11, 10, 20, 22, 32, 31 },
        { 2, 2, 2 },
        { 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1 },
        { 2, 2, 3 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 2, 2 } );

    Test( 6, numPassed,
        Flow::Create( { 2, 2, 2, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72 } ),
        Flow::Create( { 2, 2, 2, 2 }, { 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 2, 1 } ),
        { 3 }, {}, {}, {}, op,
        { 2, 1, 11, 10, 20, 22, 32, 31, 42, 41, 51, 50, 60, 62, 72, 71 },
        { 2, 2, 2, 2 },
        { 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1 },
        { 2, 2, 2, 3 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 2, 2, 2 } );

    Test( 7, numPassed,
        Flow::Create( { 2, 2, 2, 2, 5 }, { 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74 } ),
        Flow::Create( { 1, 2, 2, 2, 2 }, { 1, 0, 1, 0, 0, 2, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0 } ),
        { 4 }, {}, {}, {}, op,
        { 1, 0, 11, 10, 20, 22,32, 31, 42, 41, 50, 52, 61, 62, 71, 70 },
        { 1, 2, 2, 2, 2 },
        { 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 2, 2, 2, 5 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 2, 2, 2, 2 } );

    Flow::Print( "Test_Gather " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}