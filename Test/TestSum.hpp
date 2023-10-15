// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

static bool Test_Sum()
{
    int numPassed = 0;
    int numTests = 7;

    Flow::NArray arr = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    Flow::NArray result = Flow::Sum( arr, 0 );
    std::vector<float> expectedData = { 9, 12, 15 };
    std::vector<int> expectedShape = { 1, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_1 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_1 FAILED");

    arr = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
    result = Flow::Sum( arr, 1 );
    expectedData = { 3, 12, 21 };
    expectedShape = { 3, 1 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_2 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_2 FAILED");

    arr = Flow::Create( { 3, 2 }, { 0.2822, -0.1369, -0.0244, -1.1369, 0.4170, 0.4542 } );
    result = Flow::Sum( arr, 1 );
    expectedData = { 0.1452, -1.1613, 0.8711 };
    expectedShape = { 3, 1 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_3 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_3 FAILED");

    arr = Flow::Create( { 5 }, { -0.2245, -1.1027, -0.6163,  0.2311,  1.4139 } );
    result = Flow::Sum( arr, 0 );
    expectedData = { -0.2986 };
    expectedShape = { 1 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_4 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_4 FAILED");

    arr = Flow::Create( { 5, 2, 4 }, { 1.1207, -0.9188, -0.3266, -1.3377, 0.7320, -0.8509, 0.4469, -1.5736, 0.0684, -1.6597, 0.0162, -2.4418, -0.0988, -1.5344, 0.8390, 1.5116, 0.4829, -1.5102, -0.6237, -0.0690, -0.3929, -0.7099, 0.2078, -0.2930, 0.4350, -1.0720, -1.7251, 1.0024, -0.6134, -0.8979, 1.2491, -0.7546, 0.1477, 1.5204, 0.1064, -0.7410, 0.4673, -0.4305, 0.3669, 0.3856 } );
    result = Flow::Sum( arr, 0 );
    expectedData = { 2.2547, -3.6403, -2.5528, -3.5871, 0.0942, -4.4236,  3.1097, -0.7240 };
    expectedShape = { 1, 2, 4 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_5 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_5 FAILED");

    result = Flow::Sum( arr, 1 );
    expectedData = { 1.8527, -1.7697, 0.1203, -2.9113, -0.0304, -3.1941, 0.8552, -0.9302, 0.0900, -2.2201, -0.4159, -0.3620, -0.1784, -1.9699, -0.4760, 0.2478, 0.6150, 1.0899, 0.4733, -0.3554 };
    expectedShape = { 5, 1, 4 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_6 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_6 FAILED");

    result = Flow::Sum( arr, 2 );
    expectedData = { -1.4624, -1.2456, -4.0169, 0.7174, -1.7200, -1.1880, -1.3597, -1.0168, 1.0335,  0.7893 };
    expectedShape = { 5, 2, 1 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Sum_7 PASSED"); numPassed++; } 
    else Flow::Print("Test_Sum_7 FAILED");

    Flow::Print( "Test_Sum " + std::to_string(numPassed) + "/" + std::to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}