// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

#pragma once

using namespace std;

static void Test_Backward()
{
    int numPassed = 0;

    // Test 1
    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 5, 12, 156 } );
    Flow::NArray arr3 = Flow::Tanh(arr1);
    Flow::NArray arr4 = Flow::Transpose( arr3, 0, 1 );
    Flow::NArray arr5 = Flow::Mul( arr4, arr2 );
    Flow::NArray arr6 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr7 = Flow::MM( arr5, arr6 );
    arr7.Backpropagate();
    Flow::Print(arr1.GetGradient());
}