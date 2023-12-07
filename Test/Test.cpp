// Copyright (c) Juan M. G. de Agüero 2023

#include "Flow.h"

#include "TestAdd.hpp"
#include "TestBroadcast.hpp"
#include "TestCrossEntropy.hpp"
#include "TestExp.hpp"
#include "TestGather.hpp"
#include "TestIndex.hpp"
#include "TestLog.hpp"
#include "TestMax.hpp"
#include "TestMM.hpp"
#include "TestMul.hpp"
#include "TestPow.hpp"
#include "TestReLU.hpp"
#include "TestSum.hpp"
#include "TestTanh.hpp"
#include "TestUnsqueeze.hpp"

int main()
{
    int numPassed = 0;

    if (Test_Add())          numPassed++;
    if (Test_Broadcast())    numPassed++;
    //if (Test_CrossEntropy()) numPassed++;
    if (Test_Exp())          numPassed++;
    if (Test_Gather())       numPassed++;
    /*if (Test_Index())        numPassed++;
    if (Test_Log())          numPassed++;
    if (Test_Max())          numPassed++;
    if (Test_MM())           numPassed++;
    if (Test_Mul())          numPassed++;
    if (Test_Pow())          numPassed++;
    if (Test_ReLU())         numPassed++;
    if (Test_Sum())          numPassed++;
    if (Test_Tanh())         numPassed++;
    if (Test_Unsqueeze())    numPassed++;*/

    int numTests = 15;
    Flow::Print( "== FLOW TEST " + std::to_string(numPassed) + "/" + std::to_string(numTests) + " ==" );
}