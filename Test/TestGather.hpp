// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static bool Test_Gather()
{
    int numPassed = 0;

    Flow::NArrayCore* arr = new Flow::NArrayCore( { 3, 2 }, { 0, 1, 10, 11, 20, 21 } );
    Flow::NArrayCore* index = new Flow::NArrayCore( { 2, 2 }, { 2, 1, 1, 0 } );
    Flow::NArrayCore* result = Flow::Gather( arr, 0, index );
    vector<float> expected = { 20, 11, 10, 1 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_1 FAILED");

    arr = new Flow::NArrayCore( { 3, 2 }, { 0, 1, 10, 11, 20, 21 } );
    index = new Flow::NArrayCore( { 3, 1 }, { 1, 0, 1 } );
    result = Flow::Gather( arr, 1, index );
    expected = { 1, 10, 21 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_2 FAILED");

    arr = new Flow::NArrayCore( { 4 }, { 0, 10, 20, 30 } );
    index = new Flow::NArrayCore( { 3 }, { 3, 0, 2 } );
    result = Flow::Gather( arr, 0, index );
    expected = { 30, 0, 20 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_5 FAILED");

    arr = new Flow::NArrayCore( { 3, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22 } );
    index = new Flow::NArrayCore( { 2, 2 }, { 2, 1, 0, 2 } );
    result = Flow::Gather( arr, 1, index );
    expected = { 2, 1, 10, 12 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_6 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_6 FAILED");

    arr = new Flow::NArrayCore( { 2, 2, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 } );
    index = new Flow::NArrayCore( { 2, 2, 2 }, { 2, 1, 1, 0, 0, 2, 2, 1 } );
    result = Flow::Gather( arr, 2, index );
    expected = { 2, 1, 11, 10, 20, 22, 32, 31 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_8 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_8 FAILED");

    arr = new Flow::NArrayCore( { 2, 2, 2, 3 }, { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72 } );
    index = new Flow::NArrayCore( { 2, 2, 2, 2 }, { 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 2, 1 } );
    result = Flow::Gather( arr, 3, index );
    expected = { 2, 1, 11, 10, 20, 22, 32, 31, 42, 41, 51, 50, 60, 62, 72, 71 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_9 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_9 FAILED");

    arr = new Flow::NArrayCore({ 2, 2, 2, 2, 5 }, { 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74 } );
    index = new Flow::NArrayCore({ 1, 2, 2, 2, 2 }, { 1, 0, 1, 0, 0, 2, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0 });
    result = Flow::Gather( arr, 4, index );
    expected = { 1, 0, 11, 10, 20, 22,32, 31, 42, 41, 50, 52, 61, 62, 71, 70 };
    if ( expected == result->Get() ) { Flow::Print("Test_Gather_10 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_10 FAILED");

    int numTests = 10;
    Flow::Print( "Test_Gather " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}