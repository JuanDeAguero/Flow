// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static void Test_Backward()
{
    int numPassed = 0;

    // Test 1
    Flow::NArray arr1 = Flow::Create( { 2, 3, 1 }, { 1, 2, 3, 1, 2, 3 } );
    Flow::NArray arr2 = Flow::Create( { 1, 4 }, { 1, 2, 3, 4 } );
    Flow::NArray arr3 = Flow::Add( arr1, arr2 );
    Flow::Print(arr3);
    arr3.Backpropagate();
    Flow::Print(arr2.GetGradient());
}