// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#pragma once

namespace Flow
{
    using namespace std;

    void Print( pair< vector<int>, vector<float> > arr );

    pair< vector<int>, vector<float> > TorchAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 );
}