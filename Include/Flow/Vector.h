// Copyright (c) 2023-2025 Juan M. G. de Agüero

#pragma once

#include <vector>

namespace Flow {
    using namespace std;

    bool Equals(vector<float> vec1, vector<float> vec2, float tolerance);

    vector<int> ToInt(vector<float> vec);
    vector<int64_t> ToInt64(vector<int> vec);
}