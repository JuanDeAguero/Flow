// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Flow/Vector.h"

static bool Test( int num, int& numPassed,
    Flow::NArray arr1, Flow::NArray arr2,
    std::vector<int> intParams, std::vector<std::vector<int>> intVecParams, std::vector<float> floatParams, std::vector<Flow::NArray> arrParams,
    Flow::NArrayCore::Operation op,
    std::vector<float> expectedData, std::vector<int> expectedShape,
    std::vector<float> expectedGradData1, std::vector<int> expectedGradShape1,
    std::vector<float> expectedGradData2, std::vector<int> expectedGradShape2 )
{
    Flow::NArray result;
    std::string name;
    bool binaryOp = false;
    switch (op)
    {
        case Flow::NArrayCore::Operation::ADD: result = Flow::Add( arr1, arr2 ); name = "Add"; binaryOp = true; break;
        case Flow::NArrayCore::Operation::MUL: result = Flow::Mul( arr1, arr2 ); name = "Mul"; binaryOp = true; break;
        case Flow::NArrayCore::Operation::MM: result = Flow::MM( arr1, arr2 ); name = "MM"; binaryOp = true; break;
        case Flow::NArrayCore::Operation::POW: result = Flow::Pow( arr1, floatParams[0] ); name = "Pow"; break;
        case Flow::NArrayCore::Operation::EXP: result = Flow::Exp(arr1); name = "Exp"; break;
        case Flow::NArrayCore::Operation::TANH: result = Flow::Tanh(arr1); name = "Tanh"; break;
        case Flow::NArrayCore::Operation::RELU: result = Flow::ReLU(arr1); name = "ReLU"; break;
        case Flow::NArrayCore::Operation::LOG: result = Flow::Log(arr1); name = "Log"; break;
        case Flow::NArrayCore::Operation::SUM: result = Flow::Sum( arr1, intParams[0] ); name = "Sum"; break;
        case Flow::NArrayCore::Operation::MAX: result = Flow::Max( arr1, intParams[0] ); name = "Max"; break;
        case Flow::NArrayCore::Operation::BROADCAST: result = Flow::Broadcast( arr1, intVecParams[0] ); name = "Broadcast"; break;
        case Flow::NArrayCore::Operation::GATHER: result = Flow::Gather( arr1, intParams[0], arr2 ); name = "Gather"; binaryOp = true; break;
        case Flow::NArrayCore::Operation::UNSQUEEZE: result = Flow::Unsqueeze( arr1, intParams[0] ); name = "Unsqueeze"; break;
        case Flow::NArrayCore::Operation::INDEX: result = Flow::Index( arr1, intParams[0], arr2 ); name = "Index"; binaryOp = true; break;
        case Flow::NArrayCore::Operation::CROSSENTROPY: result = Flow::CrossEntropy( arr1, arr2 ); name = "CrossEntropy"; binaryOp = true; break;
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
        Flow::Print( "Test_" + name + "_" + std::to_string(num) + " PASSED" );
        numPassed++;
        return true;
    }
    else
    {
        Flow::Print( "Test_" + name + "_" + std::to_string(num) + " FAILED" );
        return false;
    }
}