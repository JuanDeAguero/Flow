// Copyright (c) 2023-2024 Juan M. G. de Agüero

#pragma once

using namespace std;

static int NumTests = 0;

static int NumPassed = 0;

void TestAdd( vector<int> arrShape1, vector<int> arrShape2 );

void TestBroadcast( vector<int> arrShape, vector<int> broadcastShape );

void TestExp( vector<int> arrShape );

void TestMul( vector<int> arrShape1, vector<int> arrShape2 );

void TestSum( vector<int> arrShape, int dim );

void TestPow( vector<int> arrShape, float exponent );

void TestIndex( vector<int> arrShape, int dim, NARRAY index );

void TestGather( vector<int> arrShape, int dim, NARRAY index );

void TestMax( vector<int> arrShape, int dim );

void TestBMM( vector<int> arrShape1, vector<int> arrShape2 );

void TestLog( vector<int> arrShape );

void TestTanh( vector<int> arrShape );

void TestReLU( vector<int> arrShape );

void TestProd( vector<int> arrShape, int dim );