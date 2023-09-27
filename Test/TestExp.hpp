// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

#pragma once

using namespace std;

static bool Test_Exp()
{
    int numPassed = 0;
    
    Flow::NArray arr1 = Flow::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } );
    Flow::NArray arr2 = Flow::Exp(arr1);
    vector<float> expected = { 148.4132, 1096.6332, 2980.9580, 8103.0840, 20.0855, 54.5981, 148.4132, 54.5981 };
    if ( Flow::Equals( arr2.Get(), expected, 0.01f ) ) { Flow::Print("Test_Exp_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Exp_1 FAILED");

    int numTests = 1;
    Flow::Print( "Test_Exp " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}