// Copyright (c) 2023-2025 Juan M. G. de Agüero

#pragma once

#include <memory>
#include <vector>

#include "Flow/NArray.h"

namespace Flow {
    using namespace std;

    class Module {
    public:
        Module() = default;

        virtual NARRAY Forward(NARRAY arr);

        virtual vector<NARRAY> GetParameters();

    protected:
        vector<shared_ptr<Module>> Modules;
    };

    class Linear : public Module {
    public:
        Linear() = default;
        Linear(vector<int> weightShape, vector<int> biasShape);

        static shared_ptr<Linear> Create(vector<int> weightShape, vector<int> biasShape);

        NARRAY Forward(NARRAY arr) override;

        vector<NARRAY> GetParameters() override;

    private:
        NARRAY Weight;
        NARRAY Bias;
    };

    class Convolution : public Module {
    public:
        Convolution() = default;
        Convolution(int inChannels, int outChannels, vector<int> kernel);

        static shared_ptr<Convolution> Create(int inChannels, int outChannels, vector<int> kernel);

        NARRAY Forward(NARRAY arr) override;

        vector<NARRAY> GetParameters() override;

    private:
        NARRAY Weight;
    };
}