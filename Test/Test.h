// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

using namespace std;

static int NumTests = 0;
static int NumPassed = 0;

void TestAdd( vector<int> arrShape1, vector<int> arrShape2 );

void TestBroadcast( vector<int> arrShape, vector<int> broadcastShape );

void TestExp( vector<int> arrShape );

void TestSum( vector<int> arrShape, int dim );