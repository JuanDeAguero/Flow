// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static void Test_Mul()
{
    // Test 1: ( 3, 3 ) * ( 3 )
    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    Flow::NArray arr3 = Flow::Mul( arr1, arr2 );
    vector<float> expected = { 0, 10, 200, 3, 40, 500, 6, 70, 800 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Mul_1 PASSED");
    else Flow::Print("Test_Mul_1 FAILED");
}