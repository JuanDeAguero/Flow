// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

using namespace std;

static bool Test_MM()
{
    int numPassed = 0;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::MM;

    Test( 1, numPassed,
        Flow::Create( { 2, 2 }, { 1, 2, 3, 4 } ),
        Flow::Create( { 2, 2 }, { 5, 6, 7, 8 } ), {}, {}, {}, {}, op,
        { 19, 22, 43, 50 },
        { 2, 2 },
        { 11, 15, 11, 15 },
        { 2, 2 },
        { 4, 4, 6, 6 },
        { 2, 2 } );

    Test( 2, numPassed,
        Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } ),
        Flow::Create( { 4, 2 }, { 5, 6, 5, 6, 1, 3, 7, 8 } ), {}, {}, {}, {}, op,
        { 131, 168, 68, 89 },
        { 2, 2 },
        { 11, 11, 4, 15, 11, 11, 4, 15 },
        { 2, 4 },
        { 8, 8, 11, 11, 13, 13, 13, 13 },
        { 4, 2 } );

    int numTests = 2;
    Flow::Print( "Test_MM " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}