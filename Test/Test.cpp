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

    arr1 = Flow::Create( { 2, 3, 4 }, { 1, 2, 3, 4, 4, 65, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4, 4, 5, 7, 8, 1, 2, 3, 45 } );
    NARRAY prod = Flow::Prod( arr1, 2 );
    prod->Backpropagate();
    //Flow::Print(prod);

    arr1 = Flow::Create( { 2, 3, 1, 2, 3 }, { 1, 2, 33, 4, 55, 6, 1, 2, 33, 4, 5, 65, 1, 2, 33, 41, 5, 6, 1, 2, 33, 4, 55, 6, 1, 2, 33, 4, 5, 65, 1, 2, 33, 41, 5, 6 } );
    arr2 = Flow::Create( { 5, 3, 2 }, { 1, 7, 88, 8, 9, 9, 7, 7, 8, 8, 95, 9, 7, 777, 8, 8, 9, 92, 7, 7, 855, 8, 9, 9, 7, 7, 855, 8, 9, 9 } );
    NARRAY matmul = Flow::Matmul( arr1, arr2 );
    matmul->Backpropagate();
    //Flow::Print(matmul);

    /*arr1 = Flow::Create( { 1, 1, 3, 4 }, { 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1 } );
    NARRAY unfolded = Flow::Unfold2d( arr1, { 2, 2 } );
    NARRAY folded = Flow::Fold2d( unfolded, { 1, 1, 3, 4 }, { 2, 2 } );
    folded->Backpropagate();
    Flow::Print(folded);
    Flow::Print(arr1->GetGradient());*/

    int batchSize = 1;
    int inChannels = 3;
    int outChannels = 2;
    int kernelSize = 2;
    int inSize = 6;
    NARRAY arr = Flow::Create( { batchSize, inChannels, inSize, inSize }, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } );
    NARRAY weight = Flow::Ones({ outChannels, inChannels, kernelSize, kernelSize });
    NARRAY conv = Flow::Conv2d( arr, weight );
    conv->Backpropagate();
    //Flow::Print(weight->GetGradient());

    arr = Flow::Create( { 2, 1, 5, 10 }, { 0, 7, 3, 3, 9, 1, 6, 1, 0, 6, 4, 10, 0, 10, 0, 10, 10, 3, 9, 6, 5, 5, 4, 5, 9, 8, 6, 5, 9, 1, 1, 2, 10, 3, 4, 7, 8, 3, 6, 10, 9, 6, 5, 1, 1, 6, 2, 10, 1, 7, 6, 9, 8, 5, 1, 6, 6, 1, 8, 3, 4, 8, 5, 4, 8, 7, 2, 2, 5, 10, 0, 0, 3, 8, 4, 1, 1, 2, 8, 7, 5, 9, 4, 0, 2, 10, 8, 3, 5, 5, 1, 7, 3, 9, 6, 2, 8, 9, 6, 3 } );
    NARRAY pooled = Flow::MaxPool2d( arr, { 2, 2 } );
    Flow::Print(pooled);

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