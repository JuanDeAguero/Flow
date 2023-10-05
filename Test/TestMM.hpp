// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static bool Test_MM()
{
    int numPassed = 0;

    Flow::NArray arr1 = Flow::Create( { 2, 2 }, { 1, 2, 3, 4 } );
    Flow::NArray arr2 = Flow::Create( { 2, 2 }, { 5, 6, 7, 8 } );
    Flow::NArray result = Flow::MM( arr1, arr2 );
    result.Backpropagate();
    vector<float> expected = { 19, 22, 43, 50 };
    if ( expected == result.Get() ) { Flow::Print("Test_MM_1 PASSED"); numPassed++; }
    else Flow::Print("Test_MM_1 FAILED");

    Flow::Print(arr2.GetGradient());

    arr1 = Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } );
    arr2 = Flow::Create( { 4, 2 }, { 5, 6, 5, 6, 1, 3, 7, 8 } );
    result = Flow::MM( arr1, arr2 );
    expected = { 131, 168, 68, 89 };
    if ( expected == result.Get() ) { Flow::Print("Test_MM_2 PASSED"); numPassed++; }
    else Flow::Print("Test_MM_2 FAILED");

    int numTests = 2;
    Flow::Print( "Test_MM " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}