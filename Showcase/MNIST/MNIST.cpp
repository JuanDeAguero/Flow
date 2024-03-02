// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Flow.h"

using namespace std;

static vector<float> ReadImagesMNIST( string filePath );

static vector<float> ReadLabelsMNIST( string filePath );

int main()
{
    vector<float> trainImages = ReadImagesMNIST("../train-images-idx3-ubyte");
    vector<float> testImages = ReadImagesMNIST("../t10k-images-idx3-ubyte");
    vector<float> trainLabels = ReadLabelsMNIST("../train-labels-idx1-ubyte");
    vector<float> testLabels = ReadLabelsMNIST("../t10k-labels-idx1-ubyte");

    int n = 6000;

    trainImages.resize( 784 * n );
    testImages.resize( 784 * n );
    trainLabels.resize(n);
    testLabels.resize(n);

    NARRAY xTrain( Flow::NArray::Create( { n, 784 }, trainImages ) );
    NARRAY xTest( Flow::NArray::Create( { n, 784 }, testImages ) );
    NARRAY yTrain( Flow::NArray::Create( { n }, trainLabels ) );
    NARRAY yTest( Flow::NArray::Create( { n }, testLabels ) );

    xTrain = Flow::Div( xTrain->Copy(), Flow::NArray::Create( { 1 }, { 255.0f } ) );
    xTest = Flow::Div( xTest->Copy(), Flow::NArray::Create( { 1 }, { 255.0f } ) );

    NARRAY w1( Flow::Random({ 784, 512 }) );
    NARRAY b1( Flow::Random({ 512 }) );
    NARRAY w2( Flow::Random({ 512, 256 }) );
    NARRAY b2( Flow::Random({ 256 }) );
    NARRAY w3( Flow::Random({ 256, 10 }) );
    NARRAY b3( Flow::Random({ 10 }) );

    Flow::Optimizer optimizer( { w1, b1, w2, b2, w3, b3 }, 0.0025f, 1e-8f, 0.01f );

    for ( int epoch = 0; epoch < 500; epoch++ )
    {
        NARRAY a1 = ReLU( Add( MM( xTrain, w1 ), b1 ) );
        NARRAY a2 = ReLU( Add( MM( a1, w2 ), b2 ) );
        NARRAY yPredicted = Add( MM( a2, w3 ), b3 );
        NARRAY loss = CrossEntropy( yPredicted, yTrain );
        optimizer.ZeroGrad();
        loss->Backpropagate();
        optimizer.Step();
        Flow::Print( "epoch: " + to_string( epoch + 1 ) + "  loss: " + to_string(loss->Get()[0]) +
            "  free: " + to_string(Flow::GetCUDAFreeMemory()) );
    }

    NARRAY a1 = ReLU( Add( MM( xTest, w1 ), b1 ) );
    NARRAY a2 = ReLU( Add( MM( a1, w2 ), b2 ) );
    NARRAY yPredicted = Add( MM( a2, w3 ), b3 );

    n = 100;
    int numCorrect = 0;
    for ( int i = 0; i < n; i++ )
    {
        int maxPredIndex = 0;
        float maxPredValue = 0.0f;
        for ( int j = 0; j < 10; j++ )
        {
            float value = yPredicted->Get({ i, j });
            if ( value > maxPredValue )
            {
                maxPredValue = value;
                maxPredIndex = j;
            }
        }
        if ( yTest->Get({ i }) == maxPredIndex ) numCorrect++;
    }
    Flow::Print( to_string(numCorrect) + "/" + to_string(n) );
}

vector<float> ReadImagesMNIST( string filePath )
{
    vector< vector< unsigned char > > images;
    ifstream file( filePath, ios::binary );
    if (!file.is_open()) throw runtime_error("Cannot open MNIST image file.");
    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = _byteswap_ulong(magicNumber);
    if( magicNumber != 2051 ) throw runtime_error("Invalid MNIST image file.");
    file.read( (char*)&numImages, 4 );
    numImages = _byteswap_ulong(numImages);
    file.read( (char*)&rows, 4 );
    rows = _byteswap_ulong(rows);
    file.read( (char*)&cols, 4 );
    cols = _byteswap_ulong(cols);
    for ( int i = 0; i < numImages; i++ )
    {
        vector< unsigned char > image( rows * cols );
        file.read( (char*)image.data(), rows * cols );
        images.push_back(image);
    }
    file.close();
    vector<float> imagesData;
    for ( auto image : images )
    {
        for ( auto imageChar : image ) imagesData.push_back((float)imageChar);
    }
    return imagesData;
}

vector<float> ReadLabelsMNIST( string filePath )
{
    vector< unsigned char > labels;
    ifstream file( filePath, ios::binary );
    if (!file.is_open()) throw runtime_error("Cannot open MNIST label file.");
    int magicNumber = 0, numLabels = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = _byteswap_ulong(magicNumber);
    if ( magicNumber != 2049 ) throw runtime_error("Invalid MNIST label file.");
    file.read( (char*)&numLabels, 4 );
    numLabels = _byteswap_ulong(numLabels);
    labels.resize(numLabels);
    file.read( (char*)labels.data(), numLabels );
    file.close();
    vector<float> labelsData;
    for ( auto label : labels ) labelsData.push_back((float)label);
    return labelsData;
}