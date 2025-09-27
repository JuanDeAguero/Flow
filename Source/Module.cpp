// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/Module.h"

using namespace std;

NARRAY Flow::Module::Forward(NARRAY arr) {
    return arr;
}

vector<NARRAY> Flow::Module::GetParameters() {
    vector<NARRAY> params;
    for (auto& module : Modules) {
        auto moduleParams = module->GetParameters();
        params.insert(params.end(), moduleParams.begin(), moduleParams.end());
    }
    return params;
}

Flow::Linear::Linear(vector<int> weightShape, vector<int> biasShape) {
    Weight = Random(weightShape);
    Bias = Random(biasShape);
}

shared_ptr<Flow::Linear> Flow::Linear::Create(vector<int> weightShape, vector<int> biasShape) {
    return make_shared<Linear>(weightShape, biasShape);
}

NARRAY Flow::Linear::Forward(NARRAY arr) {
    return Add(Matmul(arr, Weight), Bias);
}

vector<NARRAY> Flow::Linear::GetParameters() {
    return { Weight, Bias };
}

Flow::Convolution::Convolution(int inChannels, int outChannels, vector<int> kernel) {
    Weight = Random({ outChannels, inChannels, kernel[0], kernel[1] });
}

shared_ptr<Flow::Convolution> Flow::Convolution::Create(int inChannels, int outChannels, vector<int> kernel) {
    return make_shared<Convolution>(inChannels, outChannels, kernel);
}

NARRAY Flow::Convolution::Forward(NARRAY arr) {
    return Conv2d(arr, Weight);
}

vector<NARRAY> Flow::Convolution::GetParameters() {
    return { Weight };
}