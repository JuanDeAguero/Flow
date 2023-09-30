// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

#pragma once

using namespace std;

static bool Test_Backward()
{
    int numPassed = 0;

    Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
    Flow::NArray result = Flow::Add( arr1, arr2 );
    result.Backpropagate();
    vector<float> expectedData1 = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    vector<int> expectedShape1 = { 3, 3 };
    vector<float> expectedData2 = { 3, 3, 3 };
    vector<int> expectedShape2 = { 3 };
    if ( expectedData1 == arr1.GetGradient().Get() && expectedShape1 == arr1.GetGradient().GetShape() && expectedData2 == arr2.GetGradient().Get() && expectedShape2 == arr2.GetGradient().GetShape() ) { Flow::Print("Test_Backward_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Backward_1 FAILED");

    Flow::NArray arr = Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } );
    result = Flow::Exp(arr);
    result.Backpropagate();
    vector<float> expectedData = { 148.4132, 1096.6332, 2980.9580, 8103.0840, 20.0855, 54.5981, 148.4132, 54.5981 };
    vector<int> expectedShape = { 2, 4 };
    if ( Flow::Equals( arr.GetGradient().Get(), expectedData, 0.01f ) && expectedShape == arr.GetGradient().GetShape() ) { Flow::Print("Test_Backward_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Backward_2 FAILED");

    arr = Flow::Create( { 3, 2 }, { 0, 1, 10, 11, 20, 21 } );
    Flow::NArray index = Flow::Create( { 2, 2 }, { 2, 1, 1, 0 } );
    result = Flow::Gather( arr, 0, index );
    result.Backpropagate();
    expectedData = { 0, 1, 1, 1, 1, 0 };
    expectedShape = { 3, 2 };
    if ( Flow::Equals( arr.GetGradient().Get(), expectedData, 0.01f ) && expectedShape == arr.GetGradient().GetShape() ) { Flow::Print("Test_Gather_3 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_3 FAILED");

    arr = Flow::Create( { 3, 4 }, { 0.1427, 0.0231, -0.5414, -1.0009, -0.4664, 0.2647, -0.1228, -1.1068, -1.1734, -0.6571, 0.7230, -0.6004 } );
    index = Flow::Create( { 2 }, { 0, 2 } );
    result = Flow::Index( arr, 0, index );
    result.Backpropagate();
    expectedData = { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    expectedShape = { 3, 4 };
    if ( Flow::Equals( arr.GetGradient().Get(), expectedData, 0.01f ) && expectedShape == arr.GetGradient().GetShape() ) { Flow::Print("Test_Gather_4 PASSED"); numPassed++; }
    else Flow::Print("Test_Gather_4 FAILED");

    arr1 = Flow::Create( { 3, 3 }, { 0.6384, 0.6658, 0.6132, 0.6974, 0.8639, 0.3139, 0.6581, 0.6245, 0.7997 } );
    arr2 = Flow::Create( { 3 }, { 0.5826, 0.2585, 0.3879 } );
    result = Flow::Mul( arr1, arr2 );
    result.Backpropagate();
    expectedData1 = { 0.5826, 0.2585, 0.3879, 0.5826, 0.2585, 0.3879, 0.5826, 0.2585, 0.3879 };
    expectedShape1 = { 3, 3 };
    expectedData2 = { 1.9940, 2.1542, 1.7268 };
    expectedShape2 = { 3 };
    if ( Flow::Equals( expectedData1, arr1.GetGradient().Get(), 0.01f ) && expectedShape1 == arr1.GetGradient().GetShape() && Flow::Equals( expectedData2, arr2.GetGradient().Get(), 0.01f ) && expectedShape2 == arr2.GetGradient().GetShape() ) { Flow::Print("Test_Backward_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Backward_5 FAILED");
    
    int numTests = 5;
    Flow::Print( "Test_Backward " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}