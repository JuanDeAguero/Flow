// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

#pragma once

using namespace std;

static void Test_Add()
{
    int numPassed = 0;

    // Test 1: ( 3, 3 ) + ( 3 )
    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    Flow::NArray arr3 = Flow::Add( arr1, arr2 );
    vector<float> expected = { 1, 11, 102, 4, 14, 105, 7, 17, 108 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_1 FAILED");

    // Test 2: ( 3, 1 ) + ( 1, 3 )
    arr1 = Flow::Create( { 3, 1 }, { 0, 1, 2 } );
    arr2 = Flow::Create( { 1, 3 }, { 3, 4, 5 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 3, 4, 5, 4, 5, 6, 5, 6, 7 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_2 FAILED");

    // Test 3: ( 2, 2 ) + ( 2, 2 )
    arr1 = Flow::Create( { 2, 2 }, { 4, 1, 2, 3 } );
    arr2 = Flow::Create( { 2, 2 }, { 3, 6, 2, 5 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 7, 7, 4, 8 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_3 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_3 FAILED");

    // Test 4: ( 3 ) + ( 3 )
    arr1 = Flow::Create( { 3 }, { 1, 2, 3 } );
    arr2 = Flow::Create( { 3 }, { 4, 5, 6 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 5, 7, 9 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_4 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_4 FAILED");

    // Test 5: ( 3, 2 ) + ( 2 )
    arr1 = Flow::Create( { 3, 2 }, { 1, 2, 3, 4, 5, 6 } );
    arr2 = Flow::Create( { 2 }, { 1, 2 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 2, 4, 4, 6, 6, 8 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_5 FAILED");

    // Test 6: ( 3, 3 ) + ( 1 )
    arr1 = Flow::Create( { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 } );
    arr2 = Flow::Create( { 1 }, { 10 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_6 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_6 FAILED");

    // Test 7: ( 2 ) + ( 3, 2 )
    arr1 = Flow::Create( { 2 }, { 1, 2 } );
    arr2 = Flow::Create( { 3, 2 }, { 3, 4, 5, 6, 7, 8 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 6, 6, 8, 8, 10 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_7 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_7 FAILED");

    // Test 8: ( 2, 1 ) + ( 2, 3 )
    arr1 = Flow::Create( { 2, 1 }, { 1, 2 } );
    arr2 = Flow::Create( { 2, 3 }, { 3, 4, 5, 6, 7, 8 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 5, 6, 8, 9, 10 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_8 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_8 FAILED");

    // Test 9: ( 2 ) + ( 2 )
    arr1 = Flow::Create( { 2 }, { 1, 2 } );
    arr2 = Flow::Create( { 2 }, { 3, 4 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 6 };
    if ( expected == arr3.Get() ) { Flow::Print("Test_Add_9 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_9 FAILED");

    // Test 10: ( 3, 1 ) + ( 1, 4 )
    arr1 = Flow::Create( { 3, 1 }, { 1, 2, 3 } );
    arr2 = Flow::Create( { 1, 4 }, { 1, 2, 3, 4 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 };
    if ( expected == arr3.Get() && arr3.GetShape()[0] == 3 && arr3.GetShape()[1] == 4 && !arr3.GetShape()[2] )
         { Flow::Print("Test_Add_10 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_10 FAILED");

    // Test 11: ( 2, 3, 3 ) + ( 3 )
    arr1 = Flow::Create( { 2, 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8,  8, 7, 6, 5, 4, 3, 2, 1, 0 } );
    arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 1, 11, 102, 4, 14, 105, 7, 17, 108, 9, 17, 106, 6, 14, 103, 3, 11, 100 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_11 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_11 FAILED");

    // Test 12: ( 2, 3, 1 ) + ( 1, 1, 3 )
    arr1 = Flow::Create( { 2, 3, 1 }, { 0, 1, 2, 3, 4, 5 } );
    arr2 = Flow::Create( { 1, 1, 3 }, { 3, 4, 5 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_12 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_12 FAILED");

    // Test 13: ( 2, 2, 2 ) + ( 2, 2, 2 )
    arr1 = Flow::Create( { 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 } );
    arr2 = Flow::Create( { 2, 2, 2 }, { 8, 7, 6, 5, 4, 3, 2, 1 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 9, 9, 9, 9, 9, 9, 9, 9 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_13 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_13 FAILED");

    // Test 14: ( 2, 3, 2 ) + ( 2 )
    arr1 = Flow::Create( { 2, 3, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 } );
    arr2 = Flow::Create( { 2 }, { 1, 2 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_14 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_14 FAILED");

    // Test 15: ( 2, 2, 2, 2 ) + ( 2, 2, 2, 2 )
    arr1 = Flow::Create( { 2, 2, 2, 2 }, { 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16 } );
    arr2 = Flow::Create( { 2, 2, 2, 2 }, { 16, 14, 12, 10, 8, 6, 4, 2, 15, 13, 11, 9, 7, 5, 3, 1 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_15 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_15 FAILED");

    // Test 16: ( 2, 2, 2, 2 ) + ( 2 )
    arr1 = Flow::Create( { 2, 2, 2, 2 }, { 11, 13, 15, 17, 12, 14, 16, 18, 13, 15, 17, 19, 14, 16, 18, 20 } );
    arr2 = Flow::Create( { 2 }, { 1, 3 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 12, 16, 16, 20, 13, 17, 17, 21, 14, 18, 18, 22, 15, 19, 19, 23 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_16 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_16 FAILED");

    // Test 17: ( 5, 2, 3, 2 ) + ( 2, 3, 1 )
    arr1 = Flow::Create( { 5, 2, 3, 2 }, { 51, 53, 52, 54, 53, 55, 54, 56, 55, 57, 56, 58, 57, 59, 58, 60, 59, 61, 60, 62, 61, 63, 62, 64, 63, 65, 64, 66, 65, 67, 66, 68, 67, 69, 68, 70, 69, 71, 70, 72, 71, 73, 72, 74, 73, 75, 74, 76, 75, 77, 76, 78, 77, 79, 78, 80, 79, 81, 80, 82 } );
    arr2 = Flow::Create( { 2, 3, 1 }, { 1, 2, 3, 4, 5, 6 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 52, 54, 54, 56, 56, 58, 58, 60, 60, 62, 62, 64, 58, 60, 60, 62, 62, 64, 64, 66, 66, 68, 68, 70, 64, 66, 66, 68, 68, 70, 70, 72, 72, 74, 74, 76, 70, 72, 72, 74, 74, 76, 76, 78, 78, 80, 80, 82, 76, 78, 78, 80, 80, 82, 82, 84, 84, 86, 86, 88 };
    if ( expected == arr3.Get() )  { Flow::Print("Test_Add_17 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_17 FAILED");

    Flow::Print("Test_Add " + to_string(numPassed) + "/17");
}