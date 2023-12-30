// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <functional>
#include <vector>

#include "Flow/NArray.h"

namespace Flow
{
    using namespace std;

    class Optimizer
    {

    public:

        Optimizer( vector<reference_wrapper<NArray>> arrays, float learningRate, float epsilon, float weightDecay );

        void ZeroGrad();

        void Step();

    private:

        vector<reference_wrapper<NArray>> Arrays;

        float LearningRate;

        float Epsilon;

        float WeightDecay;

        int Time;

        vector<NArrayCore*> Beta1s;

        vector<NArrayCore*> Beta2s;

        vector<NArrayCore*> Ms;

        vector<NArrayCore*> Vs;

    };
}