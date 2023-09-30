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
    result.Backpropagate();
    vector<float> data = { -0.4076 };
    vector<int> shape = { 1, 1 };
    vector<float> dataGrad = { 0.1674, -0.1224, -0.0450, -0.0450,  0.1674, -0.1224 };
    vector<int> shapeGrad = { 2, 3 };
    if ( Flow::Equals( data, result.Get(), 0.01f ) && shape == result.GetShape() &&
        Flow::Equals( dataGrad, arr1.GetGradient().Get(), 0.01f ) && shapeGrad == arr1.GetGradient().GetShape() )
    {
        Flow::Print("Test_CrossEntropy_1 PASSED");
        numPassed++;
    }
    else Flow::Print("Test_CrossEntropy_1 FAILED");

    int numTests = 1;
    Flow::Print( "Test_CrossEntropy " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}