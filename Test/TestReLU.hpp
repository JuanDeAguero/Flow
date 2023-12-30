// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_ReLU()
{
    int numPassed = 0;
    int numTests = 4;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::RELU;

    Test( 1, numPassed,
        Flow::Create( { 2, 2, 2, 2 }, { -0.5, 1.5, -1, 2, 1.5, -1.5, 2.5, -2.5, 0, 0, 0, 0, 0, 0, 0, 0 } ), nullptr, {}, {}, {}, {}, op,
        { 0, 1.5, 0, 2, 1.5, 0, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 2, 2, 2 },
        { 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 2, 2, 2 }, {}, {} );

    Test( 2, numPassed,
        Flow::Create( { 2, 2, 2 }, { -0.5, 1.5, -1, 2, 1.5, -1.5, 2.5, -2.5 } ), nullptr, {}, {}, {}, {}, op,
        { 0, 1.5, 0, 2, 1.5, 0, 2.5, 0 },
        { 2, 2, 2 },
        { 0, 1, 0, 1, 1, 0, 1, 0 },
        { 2, 2, 2 }, {}, {} );

    Test( 3, numPassed,
        Flow::Create( { 3, 3 }, { -1, 1, 0, 2, -2, 2.5, -0.5, 0, 3 } ), nullptr, {}, {}, {}, {}, op,
        { 0, 1, 0, 2, 0, 2.5, 0, 0, 3 },
        { 3, 3 },
        { 0, 1, 0, 1, 0, 1, 0, 0, 1 },
        { 3, 3 }, {}, {} );

    Test( 4, numPassed,
        Flow::Create( { 5 }, { -1, 1, 0, 2, -2 } ), nullptr, {}, {}, {}, {}, op,
        { 0, 1, 0, 2, 0 },
        { 5 },
        { 0, 1, 0, 1, 0 },
        { 5 }, {}, {} );

    Flow::Print( "Test_ReLU " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}