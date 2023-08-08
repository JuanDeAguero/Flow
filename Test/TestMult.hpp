// Copyright (c) Juan M. G. de Agüero 2023

#include <vector>

#include "Flow/Log.h"
#include "Flow/NArray.h"

using namespace std;

static void Test_Mult()
{
    // Test 1: ( 3, 3 ) * ( 3 )
    Flow::NArray* arr1 = new Flow::NArray( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray* arr2 = new Flow::NArray( { 3 }, { 1, 10, 100 } );
    Flow::NArray* arr3 = Flow::Mult( arr1, arr2 );
    vector<float> expected = { 0, 10, 200, 3, 40, 500, 6, 70, 800 };
    if ( expected == arr3->Get() ) Flow::Log("Test_Mult_1 PASSED");
    else Flow::Log("Test_Mult_1 FAILED");
}