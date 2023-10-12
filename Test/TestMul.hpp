// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

using namespace std;

static bool Test_Mul()
{
    int numPassed = 0;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::MUL;

    Test( 1, numPassed,
        Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ),
        Flow::Create( { 3 }, { 1, 10, 100 } ), {}, {}, {}, {}, op,
        { 0, 10, 200, 3, 40, 500, 6, 70, 800 },
        { 3, 3 },
        { 1, 10, 100, 1, 10, 100, 1, 10, 100 },
        { 3, 3 },
        { 9, 12, 15 },
        { 3 } );

    int numTests = 1;
    Flow::Print( "Test_Mul " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}