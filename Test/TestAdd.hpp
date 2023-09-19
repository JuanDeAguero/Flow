// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static void Test_Add()
{
    // Test 1: ( 3, 3 ) + ( 3 )
    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    Flow::NArray arr3 = Flow::Add( arr1, arr2 );
    vector<float> expected = { 1, 11, 102, 4, 14, 105, 7, 17, 108 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_1 PASSED");
    else Flow::Print("Test_Add_1 FAILED");

    // Test 2: ( 3, 1 ) + ( 1, 3 )
    arr1 = Flow::Create( { 3, 1 }, { 0, 1, 2 } );
    arr2 = Flow::Create( { 1, 3 }, { 3, 4, 5 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 3, 4, 5, 4, 5, 6, 5, 6, 7 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_2 PASSED");
    else Flow::Print("Test_Add_2 FAILED");

    // Test 3: ( 2, 2 ) + ( 2, 2 )
    arr1 = Flow::Create( { 2, 2 }, { 4, 1, 2, 3 } );
    arr2 = Flow::Create( { 2, 2 }, { 3, 6, 2, 5 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 7, 7, 4, 8 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_3 PASSED");
    else Flow::Print("Test_Add_3 FAILED");

    // Test 4: ( 3 ) + ( 3 )
    arr1 = Flow::Create( { 3 }, { 1, 2, 3 } );
    arr2 = Flow::Create( { 3 }, { 4, 5, 6 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 5, 7, 9 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_4 PASSED");
    else Flow::Print("Test_Add_4 FAILED");

    // Test 5: ( 3, 2 ) + ( 2 )
    arr1 = Flow::Create( { 3, 2 }, { 1, 2, 3, 4, 5, 6 } );
    arr2 = Flow::Create( { 2 }, { 1, 2 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 2, 4, 4, 6, 6, 8 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_5 PASSED");
    else Flow::Print("Test_Add_5 FAILED");

    // Test 7: ( 2 ) + ( 3, 2 )
    arr1 = Flow::Create( { 2 }, { 1, 2 } );
    arr2 = Flow::Create( { 3, 2 }, { 3, 4, 5, 6, 7, 8 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 6, 6, 8, 8, 10 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_7 PASSED");
    else Flow::Print("Test_Add_7 FAILED");

    // Test 8: ( 2, 1 ) + ( 2, 3 )
    arr1 = Flow::Create( { 2, 1 }, { 1, 2 } );
    arr2 = Flow::Create( { 2, 3 }, { 3, 4, 5, 6, 7, 8 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 5, 6, 8, 9, 10 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_8 PASSED");
    else Flow::Print("Test_Add_8 FAILED");

    // Test 9: ( 2 ) + ( 2 )
    arr1 = Flow::Create( { 2 }, { 1, 2 } );
    arr2 = Flow::Create( { 2 }, { 3, 4 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 4, 6 };
    if ( expected == arr3.Get() ) Flow::Print("Test_Add_9 PASSED");
    else Flow::Print("Test_Add_9 FAILED");

    // Test 10: ( 3, 1 ) + ( 1, 4 )
    arr1 = Flow::Create( { 3, 1 }, { 1, 2, 3 } );
    arr2 = Flow::Create( { 1, 4 }, { 1, 2, 3, 4 } );
    arr3 = Flow::Add( arr1, arr2 );
    expected = { 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 };
    if ( expected == arr3.Get() && arr3.GetShape()[0] == 3 && arr3.GetShape()[1] == 4 && !arr3.GetShape()[2] )
        Flow::Print("Test_Add_10 PASSED");
    else Flow::Print("Test_Add_10 FAILED");
}