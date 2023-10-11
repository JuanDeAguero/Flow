// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test_Unsqueeze()
{
    int numPassed = 0;

    Flow::NArray arr = Flow::Create( { 3, 1, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray result = Flow::Unsqueeze( arr, 0 );
    vector<float> expectedData = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    vector<int> expectedShape = { 1, 3, 1, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Unsqueeze_1 PASSED"); numPassed++; } 
    else Flow::Print("Test_Unsqueeze_1 FAILED");

    result = Flow::Unsqueeze( arr, 1 );
    expectedShape = { 3, 1, 1, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Unsqueeze_2 PASSED"); numPassed++; } 
    else Flow::Print("Test_Unsqueeze_2 FAILED");

    result = Flow::Unsqueeze( arr, 3 );
    expectedShape = { 3, 1, 3, 1 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Unsqueeze_3 PASSED"); numPassed++; } 
    else Flow::Print("Test_Unsqueeze_3 FAILED");

    int numTests = 3;
    Flow::Print( "Test_Unsqueeze " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}