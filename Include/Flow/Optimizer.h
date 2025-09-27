// Copyright (c) 2023-2025 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"

namespace Flow {
    using namespace std;

    class Optimizer {
    public:
        Optimizer(vector<NARRAY> arrays, float learningRate, float epsilon, float weightDecay);

        void ZeroGrad();

        void Step();

    private:
        vector<NARRAY> Arrays;

        float LearningRate;
        float Epsilon;
        float WeightDecay;
        int Time;

        vector<NARRAY> Beta1s;
        vector<NARRAY> Beta2s;
        vector<NARRAY> Ms;
        vector<NARRAY> Vs;
    };
}