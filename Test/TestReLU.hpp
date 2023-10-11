// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test_ReLU()
{
    int numPassed = 0;

    Flow::NArray arr = Flow::Create( { 2, 2, 2, 2 }, { -0.5, 1.5, -1, 2, 1.5, -1.5, 2.5, -2.5, 0, 0, 0, 0, 0, 0, 0, 0 } );
    Flow::NArray result = Flow::ReLU(arr);
    result.Backpropagate();
    vector<float> data = { 0, 1.5, 0, 2, 1.5, 0, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    vector<int> shape = { 2, 2, 2, 2 };
    vector<float> dataGrad = { 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    vector<int> shapeGrad = { 2, 2, 2, 2 };
    if ( Flow::Equals( data, result.Get(), 0.01f ) && shape == result.GetShape() &&
        Flow::Equals( dataGrad, arr.GetGradient().Get(), 0.01f ) && shapeGrad == arr.GetGradient().GetShape()) { Flow::Print("Test_ReLU_1 PASSED"); numPassed++; }
    else { Flow::Print("Test_ReLU_1 FAILED"); }

    arr = Flow::Create( { 2, 2, 2 }, { -0.5, 1.5, -1, 2, 1.5, -1.5, 2.5, -2.5 } );
    result = Flow::ReLU(arr);
    data = { 0, 1.5, 0, 2, 1.5, 0, 2.5, 0 };
    shape = { 2, 2, 2 };
    if ( Flow::Equals( data, result.Get(), 0.01f ) && shape == result.GetShape() ) { Flow::Print("Test_ReLU_2 PASSED"); numPassed++; }
    else { Flow::Print("Test_ReLU_2 FAILED"); }

    arr = Flow::Create( { 3, 3 }, { -1, 1, 0, 2, -2, 2.5, -0.5, 0, 3 } );
    result = Flow::ReLU(arr);
    data = { 0, 1, 0, 2, 0, 2.5, 0, 0, 3 };
    shape = { 3, 3 };
    if ( Flow::Equals( data, result.Get(), 0.01f ) && shape == result.GetShape() ) { Flow::Print("Test_ReLU_3 PASSED"); numPassed++; }
    else { Flow::Print("Test_ReLU_3 FAILED"); }

    arr = Flow::Create( { 5 }, { -1, 1, 0, 2, -2 } );
    result = Flow::ReLU(arr);
    data = { 0, 1, 0, 2, 0 };
    shape = { 5 };
    if ( Flow::Equals( data, result.Get(), 0.01f ) && shape == result.GetShape() ) { Flow::Print("Test_ReLU_4 PASSED"); numPassed++; }
    else { Flow::Print("Test_ReLU_4 FAILED"); }

    int numTests = 4;
    Flow::Print( "Test_ReLU " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}