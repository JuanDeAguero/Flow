// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

static bool Test_Max()
{
    int numPassed = 0;
    int numTests = 1;

    Flow::NArray arr = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray result = Flow::Max( arr, 0 );
    std::vector<float> expectedData = { 6, 7, 8 };
    std::vector<int> expectedShape = { 1, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Max_1 PASSED"); numPassed++; } 
    else Flow::Print("Test_Max_1 FAILED");

    Flow::Print( "Test_Max " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}