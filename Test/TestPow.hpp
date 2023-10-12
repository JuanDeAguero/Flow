// Cop,yright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

using namespace std;

static bool Test_Pow()
{
    int numPassed = 0;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::POW;

    Test( 1, numPassed,
        Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), Flow::NArray(), {}, {},
        { 3.0f }, {}, op,
        { 0, 1, 8, 27, 64, 125, 216, 343, 512 },
        { 3, 3 },
        { 0, 3, 12, 27, 48, 75, 108, 147, 192 },
        { 3, 3 }, {}, {} );

    int numTests = 1;
    Flow::Print( "Test_Pow " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}