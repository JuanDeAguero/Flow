// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

using namespace std;

static bool Test_Broadcast()
{
    int numPassed = 0;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::BROADCAST;

    Test( 1, numPassed,
        Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), Flow::NArray(), {},
        { { 1, 3, 3 } }, {}, {}, op,
        { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
        { 1, 3, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 3 }, {}, {} );

    Test( 2, numPassed,
        Flow::Create( { 3 }, { 1, 2, 3 } ), Flow::NArray(), {},
        { { 3, 3 } }, {}, {}, op,
        { 1, 2, 3, 1, 2, 3, 1, 2, 3 },
        { 3, 3 },
        { 3, 3, 3 },
        { 3 }, {}, {} );

    Test( 3, numPassed,
        Flow::Create( { 3 }, { 4, 5, 6 } ), Flow::NArray(), {},
        { { 3, 3 } }, {}, {}, op,
        { 4, 5, 6, 4, 5, 6, 4, 5, 6 },
        { 3, 3 },
        { 3, 3, 3 },
        { 3 }, {}, {}
    );
    
    Test( 4, numPassed,
        Flow::Create( { 1, 3 }, { 7, 8, 9 } ), Flow::NArray(), {},
        { { 4, 1, 3 } }, {}, {}, op,
        { 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9 },
        { 4, 1, 3 },
        { 4, 4, 4 },
        { 1, 3 }, {}, {}
    );

    Test( 5, numPassed,
        Flow::Create( { 1 }, { 10 } ), Flow::NArray(), {},
        { { 3, 3, 3 } }, {}, {}, op,
        { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 },
        { 3, 3, 3 },
        { 27 },
        { 1 }, {}, {}
    );

    Test( 6, numPassed,
        Flow::Create( { 1, 1, 1, 3 }, { 13, 14, 15 } ), Flow::NArray(), {},
        { { 4, 4, 4, 3 } }, {}, {}, op,
        { 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15 },
        { 4, 4, 4, 3 },
        { 64, 64, 64 },
        { 1, 1, 1, 3 }, {}, {}
    );

    Test( 7, numPassed,
        Flow::Create( { 1, 1, 1, 1, 3 }, { 22, 23, 24 } ), Flow::NArray(), {},
        { { 2, 2, 2, 2, 3 } }, {}, {}, op,
        { 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24 },
        { 2, 2, 2, 2, 3 },
        { 16, 16, 16 },
        { 1, 1, 1, 1, 3 }, {}, {}
    );

    int numTests = 7;
    Flow::Print( "Test_Broadcast " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}