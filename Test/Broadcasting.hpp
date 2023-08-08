// Copyright (c) Juan M. G. de Agüero 2023

#include <vector>

#include "Flow/Log.h"
#include "Flow/NArray.h"

using namespace std;

void Test_Broadcasting()
{
    // Test 1: ( 3, 3 ) + ( 3 )
    Flow::NArray* arr1 = new Flow::NArray( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray* arr2 = new Flow::NArray( { 3 }, { 1, 10, 100 } );
    Flow::NArray* arr3 = Flow::Add( arr1, arr2 );
    vector<float> expected = { 1, 11, 102, 4, 14, 105, 7, 17, 108 };
    bool passed = true;
    for ( int i = 0; i < expected.size(); i++ )
    {
        if ( arr3->Get()[i] != expected[i] )
        {
            passed = false;
            break;
        }
    }
    if (passed) Flow::Log("Test_Broadcasting_1 PASSED");
    else Flow::Log("Test_Broadcasting_1 FAILED");

    // Test 2: ( 3, 1 ) + ( 1, 3 )
    // TODO
}