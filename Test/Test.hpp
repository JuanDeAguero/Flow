// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

using namespace std;

static bool Test( int num, int& numPassed, NARRAY arr1, NARRAY arr2, vector<int> intParams,
    vector<vector<int>> intVecParams, vector<float> floatParams, vector<NARRAY> arrParams,
    Flow::NArray::Operation op, vector<float> expectedData, vector<int> expectedShape,
    vector<float> expectedGradData1, vector<int> expectedGradShape1,
    vector<float> expectedGradData2, vector<int> expectedGradShape2 )
{
    NARRAY result;
    string name;
    bool binaryOp = false;
    switch (op)
    {
        case Flow::NArray::Operation::ADD:
            result = Flow::Add( arr1, arr2 );
            name = "Add";
            binaryOp = true;
            break;
        case Flow::NArray::Operation::MUL:
            result = Flow::Mul( arr1, arr2 );
            name = "Mul";
            binaryOp = true;
            break;
        case Flow::NArray::Operation::MM:
            result = Flow::MM( arr1, arr2 );
            name = "MM";
            binaryOp = true;
            break;
        case Flow::NArray::Operation::POW:
            result = Flow::Pow( arr1, floatParams[0] );
            name = "Pow";
            break;
        case Flow::NArray::Operation::EXP:
            result = Flow::Exp(arr1);
            name = "Exp";
            break;
        case Flow::NArray::Operation::TANH:
            result = Flow::Tanh(arr1);
            name = "Tanh";
            break;
        case Flow::NArray::Operation::RELU:
            result = Flow::ReLU(arr1);
            name = "ReLU";
            break;
        case Flow::NArray::Operation::LOG:
            result = Flow::Log(arr1);
            name = "Log";
            break;
        case Flow::NArray::Operation::SUM:
            result = Flow::Sum( arr1, intParams[0] );
            name = "Sum";
            break;
        case Flow::NArray::Operation::MAX:
            result = Flow::Max( arr1, intParams[0] );
            name = "Max";
            break;
        case Flow::NArray::Operation::BROADCAST:
            result = Flow::Broadcast( arr1, intVecParams[0] );
            name = "Broadcast";
            break;
        case Flow::NArray::Operation::GATHER:
            result = Flow::Gather( arr1, intParams[0], arr2 );
            name = "Gather";
            binaryOp = true;
            break;
        case Flow::NArray::Operation::UNSQUEEZE:
            result = Flow::Unsqueeze( arr1, intParams[0] );
            name = "Unsqueeze";
            break;
        case Flow::NArray::Operation::INDEX:
            result = Flow::Index( arr1, intParams[0], arr2 );
            name = "Index";
            binaryOp = true;
            break;
        case Flow::NArray::Operation::CROSSENTROPY:
            result = Flow::CrossEntropy( arr1, arr2 );
            name = "CrossEntropy";
            binaryOp = true;
            break;
    }
    result->Backpropagate();
    bool passed = false;
    if ( Flow::Equals( expectedData, result->Get(), 0.01f ) &&
        expectedShape == result->GetShape() &&
        Flow::Equals( expectedGradData1, arr1->GetGradient()->Get(), 0.01f ) &&
        expectedGradShape1 == arr1->GetGradient()->GetShape() )
    {
        if (binaryOp)
        {
            if ( Flow::Equals( expectedGradData2, arr2->GetGradient()->Get(), 0.01f ) &&
                expectedGradShape2 == arr2->GetGradient()->GetShape() )
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