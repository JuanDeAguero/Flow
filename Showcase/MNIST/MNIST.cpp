// Copyright (c) 2023-2025 Juan M. G. de Agüero

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Flow.h"

using namespace std;

static vector<float> ReadImagesMNIST(string filePath);
static vector<float> ReadLabelsMNIST(string filePath);

class Network : public Flow::Module {
public:
    shared_ptr<Flow::Linear> Linear1, Linear2, Linear3;

    Network() {
        Linear1 = Flow::Linear::Create({ 784, 512 }, { 512 });
        Linear2 = Flow::Linear::Create({ 512, 256 }, { 256 });
        Linear3 = Flow::Linear::Create({ 256, 10 }, { 10 });
        Modules = { Linear1, Linear2, Linear3 };
    }

    NARRAY Forward(NARRAY arr) override {
        NARRAY a1 = Flow::Reshape(arr, { arr->GetShape()[0], 28 * 28 });
        NARRAY a2 = Flow::ReLU(Linear1->Forward(a1));
        NARRAY a3 = Flow::ReLU(Linear2->Forward(a2));
        return Linear3->Forward(a3);
    }
};

class CNN : public Flow::Module {
public:
    shared_ptr<Flow::Convolution> Conv1, Conv2;
    shared_ptr<Flow::Linear> Linear1, Linear2;

    CNN() {
        Conv1 = Flow::Convolution::Create(1, 10, { 5, 5 });
        Conv2 = Flow::Convolution::Create(10, 20, { 5, 5 });
        Linear1 = Flow::Linear::Create({ 320, 50 }, { 50 });
        Linear2 = Flow::Linear::Create({ 50, 10 }, { 10 });
        Modules = { Conv1, Conv2, Linear1, Linear2 };
    }

    NARRAY Forward(NARRAY arr) override {
        NARRAY a1 = Flow::Unsqueeze(arr, 1);
        NARRAY a2 = Flow::ReLU(MaxPool2d(Conv1->Forward(a1), { 2, 2 }));
        NARRAY a3 = Flow::ReLU(MaxPool2d(Conv2->Forward(a2), { 2, 2 }));
        NARRAY a4 = Flow::Reshape(a3, { a3->GetShape()[0], 320 });
        NARRAY a5 = Flow::ReLU(Linear1->Forward(a4));
        NARRAY a6 = Linear2->Forward(a5);
        return Flow::Softmax(a6, 1);
    }
};

int main() {
    vector<float> trainImages = ReadImagesMNIST("../train-images-idx3-ubyte");
    vector<float> testImages = ReadImagesMNIST("../t10k-images-idx3-ubyte");
    vector<float> trainLabels = ReadLabelsMNIST("../train-labels-idx1-ubyte");
    vector<float> testLabels = ReadLabelsMNIST("../t10k-labels-idx1-ubyte");

    int numTrain = 60000;
    int numTest = 10000;
    trainImages.resize(784 * numTrain);
    testImages.resize(784 * numTest);
    trainLabels.resize(numTrain);
    testLabels.resize(numTest);

    NARRAY xTrain(Flow::Create({ numTrain, 28, 28 }, trainImages));
    NARRAY xTest(Flow::Create({ numTest, 28, 28 }, testImages));
    NARRAY yTrain(Flow::Create({ numTrain }, trainLabels));
    NARRAY yTest(Flow::Create({ numTest }, testLabels));
    xTrain = Flow::Div(xTrain->Copy(), Flow::Create({ 1 }, { 255.0f }));
    xTest = Flow::Div(xTest->Copy(), Flow::Create({ 1 }, { 255.0f }));

    CNN network;
    Flow::Optimizer optimizer(network.GetParameters(), 0.001f, 1e-8f, 0.0f);

    for (int epoch = 0; epoch < 10; epoch++) {
        auto batches = Flow::CreateBatches(xTrain, yTrain, 100);
        for (auto batch : batches) {
            NARRAY yPredicted = network.Forward(batch.first);
            NARRAY loss = Flow::CrossEntropy(yPredicted, batch.second);
            optimizer.ZeroGrad();
            loss->Backpropagate();
            optimizer.Step();
            Flow::Print("epoch: " + to_string(epoch + 1) + "  loss: " + to_string(loss->Get()[0]) + "  free: " + to_string(Flow::CUDA_GetFreeMemory()));
        }
        
        batches = Flow::CreateBatches(xTest, yTest, 100);
        int numCorrect = 0;
        for (auto batch : batches) {
            NARRAY yPredicted = network.Forward(batch.first);
            for (int i = 0; i < batch.first->GetShape()[0]; i++) {
                int maxPredIndex = 0;
                float maxPredValue = 0.0f;
                for (int j = 0; j < 10; j++) {
                    float value = yPredicted->Get({ i, j });
                    if (value > maxPredValue) {
                        maxPredValue = value;
                        maxPredIndex = j;
                    }
                }
                Flow::Print(to_string((int)batch.second->Get({ i })) + " | " + to_string(maxPredIndex));
                if (batch.second->Get({ i }) == maxPredIndex) numCorrect++;
            }
        }
        Flow::Print(to_string(numCorrect) + "/" + to_string(numTest) + " " + to_string(((float)numCorrect / (float)numTest) * 100.0f) + "%");
    }
}

vector<float> ReadImagesMNIST(string filePath) {
    vector< vector< unsigned char > > images;
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open MNIST image file.");
    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read((char*)&magicNumber, 4);
    magicNumber = _byteswap_ulong(magicNumber);
    if(magicNumber != 2051) throw runtime_error("Invalid MNIST image file.");
    file.read((char*)&numImages, 4);
    numImages = _byteswap_ulong(numImages);
    file.read((char*)&rows, 4);
    rows = _byteswap_ulong(rows);
    file.read((char*)&cols, 4);
    cols = _byteswap_ulong(cols);
    for (int i = 0; i < numImages; i++) {
        vector< unsigned char > image(rows * cols);
        file.read((char*)image.data(), rows * cols);
        images.push_back(image);
    }
    file.close();
    vector<float> imagesData;
    for (auto image : images)
    for (auto imageChar : image) imagesData.push_back((float)imageChar);
    return imagesData;
}

vector<float> ReadLabelsMNIST(string filePath) {
    vector< unsigned char > labels;
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open MNIST label file.");
    int magicNumber = 0, numLabels = 0;
    file.read((char*)&magicNumber, 4);
    magicNumber = _byteswap_ulong(magicNumber);
    if (magicNumber != 2049) throw runtime_error("Invalid MNIST label file.");
    file.read((char*)&numLabels, 4);
    numLabels = _byteswap_ulong(numLabels);
    labels.resize(numLabels);
    file.read((char*)labels.data(), numLabels);
    file.close();
    vector<float> labelsData;
    for (auto label : labels) labelsData.push_back((float)label);
    return labelsData;
}