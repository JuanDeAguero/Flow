// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_CrossEntropy()
{
    int numPassed = 0;
    int numTests = 1;
    Flow::NArrayCore::Operation op = Flow::NArrayCore::Operation::CROSSENTROPY;

    Test( 1, numPassed,
        Flow::Create( { 2, 3 }, { 1.5, 0.5, -0.5, -0.5, 1.5, 0.5 } ),
        Flow::Create( { 2 }, { 0, 1 } ), {}, {}, {}, {}, op,
        { 0.4076 },
        { 1, 1 },
        { -0.1674, 0.1224, 0.0450, 0.0450, -0.1674, 0.1224 },
        { 2, 3 },
        { 0, 0 },
        { 2 } );

    Flow::Print( "Test_CrossEntropy " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}