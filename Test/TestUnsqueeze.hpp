// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Unsqueeze()
{
    int numPassed = 0;
    int numTests = 3;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::UNSQUEEZE;

    Test( 1, numPassed,
        Flow::Create( { 3, 1, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), nullptr,
        { 0 }, {}, {}, {}, op,
        { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
        { 1, 3, 1, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 1, 3 }, {}, {} );

    Test( 2, numPassed,
        Flow::Create( { 3, 1, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), nullptr,
        { 1 }, {}, {}, {}, op,
        { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
        { 3, 1, 1, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 1, 3 }, {}, {} );

    Test( 3, numPassed,
        Flow::Create( { 3, 1, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ), nullptr,
        { 3 }, {}, {}, {}, op,
        { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
        { 3, 1, 3, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 1, 3 }, {}, {} );

    Flow::Print( "Test_Unsqueeze " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}