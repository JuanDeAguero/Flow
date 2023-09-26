// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

#pragma once

using namespace std;

static bool Test_Broadcast()
{
    int numPassed = 0;

    Flow::NArray arr = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray result = Flow::Broadcast( arr, { 1, 3, 3 } );
    vector<float> expectedData = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    vector<int> expectedShape = { 1, 3, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_1 FAILED");

    arr = Flow::Create( { 3 }, { 1, 2, 3 } );
    result = Flow::Broadcast( arr, { 3, 3 } );
    expectedData = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    expectedShape = { 3, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_2 FAILED");

    arr = Flow::Create( { 3 }, { 4, 5, 6 } );
    result = Flow::Broadcast( arr, { 3, 3 } );
    expectedData = { 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    expectedShape = { 3, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_3 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_3 FAILED");
    
    arr = Flow::Create( { 1, 3 }, { 7, 8, 9 } );
    result = Flow::Broadcast( arr, { 4, 1, 3 } );
    expectedData = { 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9 };
    expectedShape = { 4, 1, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_4 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_4 FAILED");
    
    arr = Flow::Create( { 1 }, { 10 } );
    result = Flow::Broadcast( arr, { 3, 3, 3 } );
    expectedData = { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
    expectedShape = { 3, 3, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_5 FAILED");
    
    arr = Flow::Create( { 2 }, { 11, 12 } );
    try { result = Flow::Broadcast( arr, { 3, 3 } ); Flow::Print("Test_Broadcast_6 FAILED"); }
    catch( exception e ) { Flow::Print("Test_Broadcast_6 PASSED"); numPassed++; }

    arr = Flow::Create( { 1, 1, 1, 3 }, { 13, 14, 15 } );
    result = Flow::Broadcast( arr, { 4, 4, 4, 3 } );
    expectedData =  { 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15 };
    expectedShape = { 4, 4, 4, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_7 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_7 FAILED");

    arr = Flow::Create( { 1, 2, 1, 3 }, { 16, 17, 18, 19, 20, 21 } );
    try { result = Flow::Broadcast( arr, { 4, 5, 4, 3 } ); Flow::Print("Test_Broadcast_8 FAILED"); }
    catch( exception e ) { Flow::Print("Test_Broadcast_8 PASSED"); numPassed++; }

    arr = Flow::Create( { 1, 1, 1, 1, 3 }, { 22, 23, 24 } );
    result = Flow::Broadcast( arr, { 2, 2, 2, 2, 3 } );
    expectedData =  { 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24, 22, 23, 24 };
    expectedShape = { 2, 2, 2, 2, 3 };
    if ( expectedData == result.Get() && result.GetShape() == expectedShape ) { Flow::Print("Test_Broadcast_9 PASSED"); numPassed++; }
    else Flow::Print("Test_Broadcast_9 FAILED");

    int numTests = 9;
    Flow::Print("Test_Broadcast " + to_string(numPassed) + "/" + to_string(numTests));
    if ( numPassed == numTests ) return true;
}