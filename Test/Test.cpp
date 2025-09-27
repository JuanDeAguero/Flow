// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include <chrono>

#include "Flow.h"
#include "Test.h"

int main() {
    NumTests = 1; NumPassed = 0;
    TestAdd({ 3 }, { 3, 3 });
    Flow::Print("Test_Add " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 8; NumPassed = 0;
    TestBroadcast({ 3 }, { 3, 3 });
    TestBroadcast({ 1, 4, 10 }, { 3, 5, 4, 10 });
    TestBroadcast({ 4, 4, 5 }, { 4, 4, 5 });
    TestBroadcast({ 3, 3 }, { 1, 3, 3 });
    TestBroadcast({ 1 }, { 3, 3, 3 });
    TestBroadcast({ 1, 1, 1, 3 }, { 4, 4, 4, 3 });
    TestBroadcast({ 2, 1, 3 }, { 2, 5, 3 });
    TestBroadcast({ 3, 1, 1, 2, 1, 3 }, { 2, 3, 10, 5, 2, 5, 3 });
    Flow::Print("Test_Broadcast " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestExp({ 3, 3 });
    Flow::Print("Test_Exp " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestMul({ 3 }, { 3, 3 });
    Flow::Print("Test_Mul " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestSum({ 3, 3 }, 0);
    Flow::Print("Test_Sum " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestPow({ 3, 3 }, 3.0f);
    Flow::Print("Test_Pow " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestIndex({ 3, 4 }, 0, Flow::Create({ 2 }, { 0, 2 }));
    Flow::Print("Test_Index " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 2; NumPassed = 0;
    TestGather({ 3, 2 }, 1, Flow::Create({ 3, 1 }, { 1, 0, 1 }));
    TestGather({ 2, 2, 3 }, 2, Flow::Create({ 2, 2, 2 }, { 2, 1, 1, 0, 0, 2, 2, 1 }));
    Flow::Print("Test_Gather " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestMax({ 3, 3 }, 0);
    Flow::Print("Test_Max " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestBMM({ 10, 3, 4 }, { 10, 4, 5 });
    Flow::Print("Test_BMM " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestLog({ 3, 3 });
    Flow::Print("Test_Log " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestTanh({ 5 });
    Flow::Print("Test_Tanh " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestReLU({ 3, 2, 4 });
    Flow::Print("Test_ReLU " + to_string(NumPassed) + "/" + to_string(NumTests));

    NumTests = 1; NumPassed = 0;
    TestProd({ 3, 3 }, 0);
    Flow::Print("Test_Prod " + to_string(NumPassed) + "/" + to_string(NumTests));
}

void TestAdd(vector<int> arrShape1, vector<int> arrShape2) {
    NARRAY arr1 = Flow::RandomUniform(arrShape1, -0.9999f, 0.9999f);
    NARRAY arr2 = Flow::RandomUniform(arrShape2, -0.9999f, 0.9999f);
    NARRAY result = Flow::Add(arr1, arr2);
    result->Backpropagate();
    NARRAY grad1 = arr1->GetGradient();
    NARRAY grad2 = arr2->GetGradient();
}

void TestBroadcast(vector<int> arrShape, vector<int> broadcastShape) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Broadcast(arr, broadcastShape);
    auto start = chrono::high_resolution_clock::now();
    result->Backpropagate();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    Flow::Print(duration);
    NARRAY grad = arr->GetGradient();
}

void TestExp(vector<int> arrShape) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Exp(arr);
    result->Backpropagate();
}

void TestMul(vector<int> arrShape1, vector<int> arrShape2) {
    NARRAY arr1 = Flow::RandomUniform(arrShape1, -0.9999f, 0.9999f);
    NARRAY arr2 = Flow::RandomUniform(arrShape2, -0.9999f, 0.9999f);
    NARRAY result = Flow::Mul(arr1, arr2);
    result->Backpropagate();
    NARRAY grad1 = arr1->GetGradient();
    NARRAY grad2 = arr2->GetGradient();
}

void TestSum(vector<int> arrShape, int dim) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Sum(arr, dim);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestPow(vector<int> arrShape, float exponent) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Pow(arr, exponent);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestIndex(vector<int> arrShape, int dim, NARRAY index) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Index(arr, dim, index);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestGather(vector<int> arrShape, int dim, NARRAY index) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Gather(arr, dim, index);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestMax(vector<int> arrShape, int dim) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Max(arr, dim);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestBMM(vector<int> arrShape1, vector<int> arrShape2) {
    NARRAY arr1 = Flow::RandomUniform(arrShape1, -0.9999f, 0.9999f);
    NARRAY arr2 = Flow::RandomUniform(arrShape2, -0.9999f, 0.9999f);
    NARRAY result = Flow::BMM(arr1, arr2);
    result->Backpropagate();
    NARRAY grad1 = arr1->GetGradient();
    NARRAY grad2 = arr2->GetGradient();
}

void TestLog(vector<int> arrShape) {
    NARRAY arr = Flow::RandomUniform(arrShape, 0.0001f, 0.9999f);
    NARRAY result = Flow::Log(arr);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestTanh(vector<int> arrShape) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Tanh(arr);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestReLU(vector<int> arrShape) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::ReLU(arr);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}

void TestProd(vector<int> arrShape, int dim) {
    NARRAY arr = Flow::RandomUniform(arrShape, -0.9999f, 0.9999f);
    NARRAY result = Flow::Prod(arr, dim);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
}