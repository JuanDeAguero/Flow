// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

#pragma once

using namespace std;

static bool Test_Sum()
{
    int numPassed = 0;

    int numTests = 1;
    Flow::Print( "Test_Sum " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}