// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

#pragma once

using namespace std;

static void Test_MM()
{
    int numPassed = 0;

    // Test 1
    Flow::NArray arr1 = Flow::Create( { 2, 2 }, { 1, 2, 3, 4 } );
    Flow::NArray arr2 = Flow::Create( { 2, 2 }, { 5, 6, 7, 8 } );
    Flow::NArray arr3 = Flow::MM( arr1, arr2 );
    vector<float> expected = { 19, 22, 43, 50 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_MM_1 PASSED"); numPassed++; }
    else Flow::Print("Test_MM_1 FAILED");

    // Test 2
    arr1 = Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } );
    arr2 = Flow::Create( { 4, 2 }, { 5, 6, 5, 6, 1, 3, 7, 8 } );
    arr3 = Flow::MM( arr1, arr2 );
    expected = { 131, 168, 68, 89 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_MM_2 PASSED"); numPassed++; }
    else Flow::Print("Test_MM_2 FAILED");

    arr3.Backpropagate();
    Flow::Print(arr1.GetGradient());
}