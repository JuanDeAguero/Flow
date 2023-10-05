// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test_Max()
{
    int numPassed = 0;

    Flow::NArray arr = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray result = Flow::Max( arr, 0 );
    vector<float> expectedData = { 6, 7, 8 };
    vector<int> expectedShape = { 1, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Max_1 PASSED"); numPassed++; } 
    else Flow::Print("Test_Max_1 FAILED");

    int numTests = 1;
    Flow::Print( "Test_Max " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}