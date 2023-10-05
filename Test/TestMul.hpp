// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static bool Test_Mul()
{
    int numPassed = 0;

    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    Flow::NArray result = Flow::Mul( arr1, arr2 );
    vector<float> expected = { 0, 10, 200, 3, 40, 500, 6, 70, 800 };
    if ( expected == result.Get() ) { Flow::Print("Test_Mul_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Mul_1 FAILED");

    int numTests = 1;
    Flow::Print( "Test_Mul " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}