// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test_Exp()
{
    int numPassed = 0;
    
    Flow::NArray arr = Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } );
    Flow::NArray result = Flow::Exp(arr);
    vector<float> expectedData = { 148.4132, 1096.6332, 2980.9580, 8103.0840, 20.0855, 54.5981, 148.4132, 54.5981 };
    vector<int> expectedShape = { 2, 4 };
    if ( Flow::Equals( result.Get(), expectedData, 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Exp_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_1 FAILED");

    arr = Flow::Create( { 3 }, { 1, 2, 3 } );
    result = Flow::Exp(arr);
    expectedData = { 2.7183, 7.3891, 20.0855 };
    expectedShape = { 3 };
    if ( Flow::Equals( result.Get(), expectedData, 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Exp_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_2 FAILED");
    
    arr = Flow::Create( { 2, 2 }, { 0, -1, -2, -3 } );
    result = Flow::Exp(arr);
    expectedData = { 1.0, 0.3679, 0.1353, 0.0498 };
    expectedShape = { 2, 2 };
    if ( Flow::Equals( result.Get(), expectedData, 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Exp_3 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_3 FAILED");
    
    arr = Flow::Create( { 1, 4 }, { -1, 0, 1, 2 } );
    result = Flow::Exp(arr);
    expectedData = { 0.3679, 1.0, 2.7183, 7.3891 };
    expectedShape = { 1, 4 };
    if ( Flow::Equals( result.Get(), expectedData, 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Exp_4 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_4 FAILED");

    arr = Flow::Create( { 2, 2, 2, 2 }, { 0, 1, -1, 2, 1.5, -1.5, 2.5, -2.5 } );
    result = Flow::Exp(arr);
    expectedData = { 1.0, 2.7183, 0.3679, 7.3891, 4.4817, 0.2231, 12.1825, 0.0821 };
    expectedShape = { 2, 2, 2, 2 };
    if ( Flow::Equals( result.Get(), expectedData, 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_Exp_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_5 FAILED");

    int numTests = 5;
    Flow::Print( "Test_Exp " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}