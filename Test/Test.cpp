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

using namespace std;

int main()
{
    NARRAY arr1 = Flow::Create( { 2, 2, 3 }, { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 } );
    NARRAY arr2 = Flow::Create( { 2, 3, 2 }, { 77, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12 } );
    NARRAY bmm = Flow::BMM( arr1, arr2 );
    bmm->Backpropagate();
    //Flow::Print(arr2->GetGradient());

    NARRAY arr3 = Flow::Create( { 2, 3, 4 }, { 1, 2, 3, 4, 4, 65, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4, 4, 5, 7, 8, 1, 2, 3, 45 } );
    NARRAY prod = Flow::Prod( arr3, 2 );
    prod->Backpropagate();
    //Flow::Print(arr3->GetGradient());

    NARRAY arr4 = Flow::Create( { 2, 3, 1, 2, 3 }, { 1, 2, 33, 4, 55, 6, 1, 2, 33, 4, 5, 65, 1, 2, 33, 41, 5, 6, 1, 2, 33, 4, 55, 6, 1, 2, 33, 4, 5, 65, 1, 2, 33, 41, 5, 6 } );
    NARRAY arr5 = Flow::Create( { 5, 3, 2 }, { 1, 7, 88, 8, 9, 9, 7, 7, 8, 8, 95, 9, 7, 777, 8, 8, 9, 92, 7, 7, 855, 8, 9, 9, 7, 7, 855, 8, 9, 9 } );
    NARRAY arr6 = Flow::Matmul( arr4, arr5 );
    arr6->Backpropagate();
    Flow::Print(arr6);

    int numPassed = 0;

    if (Test_Add())          numPassed++;
    if (Test_Broadcast())    numPassed++;
    if (Test_CrossEntropy()) numPassed++;
    if (Test_Exp())          numPassed++;
    if (Test_Gather())       numPassed++;
    if (Test_Index())        numPassed++;
    if (Test_Log())          numPassed++;
    if (Test_Max())          numPassed++;
    if (Test_MM())           numPassed++;
    if (Test_Mul())          numPassed++;
    if (Test_Pow())          numPassed++;
    if (Test_ReLU())         numPassed++;
    if (Test_Sum())          numPassed++;
    if (Test_Tanh())         numPassed++;
    if (Test_Unsqueeze())    numPassed++;

    int numTests = 15;
    Flow::Print( "== FLOW TEST " + to_string(numPassed) + "/" + to_string(numTests) + " ==" );
}