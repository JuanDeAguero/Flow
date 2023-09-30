// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

#pragma once

using namespace std;

static bool Test_CrossEntropy()
{
    int numPassed = 0;

    Flow::NArray arr1 = Flow::Create( { 2, 3 }, { 1.5, 0.5, -0.5, -0.5, 1.5, 0.5 } );
    Flow::NArray arr2 = Flow::Create( { 2 }, { 0, 1 } );
    Flow::NArray result = Flow::CrossEntropy( arr1, arr2 );
    vector<float> expectedData = { -0.4076 };
    vector<int> expectedShape = { 1, 1 };
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) && expectedShape == result.GetShape() ) { Flow::Print("Test_CrossEntropy_1 PASSED"); numPassed++; }
    else Flow::Print("Test_CrossEntropy_1 FAILED");

    int numTests = 1;
    Flow::Print( "Test_CrossEntropy " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}