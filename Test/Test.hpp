// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test( int num, int& numPassed,
    Flow::NArray arr1, Flow::NArray arr2,
    vector<int> intParams, vector<vector<int>> intVecParams, vector<float> floatParams, vector<Flow::NArray> arrParams,
    Flow::NArrayCore::Operation op,
    vector<float> expectedData, vector<int> expectedShape,
    vector<float> expectedGradData1, vector<int> expectedGradShape1,
    vector<float> expectedGradData2, vector<int> expectedGradShape2 )
{
    Flow::NArray result;
    string name;
    bool binaryOp = false;
    switch (op)
    {
        case Flow::NArrayCore::Operation::ADD:
        {
            result = Flow::Add( arr1, arr2 );
            name = "Add";
            binaryOp == true;
            break;
        }
        case Flow::NArrayCore::Operation::MUL:
        {
            result = Flow::Mul( arr1, arr2 );
            name = "Mul";binaryOp == true;
            break;
        }
        case Flow::NArrayCore::Operation::MM:
        {
            result = Flow::MM( arr1, arr2 );
            name = "MM";
            binaryOp == true;
            break;
        }
        case Flow::NArrayCore::Operation::POW:
        {
            result = Flow::Pow( arr1, floatParams[0] );
            name = "Pow";
            break;
        }
        case Flow::NArrayCore::Operation::EXP:
        {
            result = Flow::Exp(arr1);
            name = "Exp";
            break;
        }
        case Flow::NArrayCore::Operation::BROADCAST:
        {
            result = Flow::Broadcast( arr1, intVecParams[0] );
            name = "Broadcast"; 
            break;
        }
    }
    result.Backpropagate();
    bool passed = false;
    if ( Flow::Equals( expectedData, result.Get(), 0.01f ) &&
        expectedShape == result.GetShape() &&
        Flow::Equals( expectedGradData1, arr1.GetGradient().Get(), 0.01f ) &&
        expectedGradShape1 == arr1.GetGradient().GetShape() )
    {
        if (binaryOp)
        {
            if ( Flow::Equals( expectedGradData2, arr2.GetGradient().Get(), 0.01f ) &&
                expectedGradShape2 == arr2.GetGradient().GetShape() )
                passed = true;
        }
        else passed = true;
    }
    if (passed)
    {
        Flow::Print( "Test_" + name + "_" + to_string(num) + " PASSED" );
        numPassed++;
        return true;
    }
    else
    {
        Flow::Print( "Test_" + name + "_" + to_string(num) + " FAILED" );
        return false;
    }
}