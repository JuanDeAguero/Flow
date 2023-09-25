// Copyright (c) 2023 Juan M. G. de Agüero

#include <fstream>
#include <iostream>
#include <vector>

#include "Flow.h"

#pragma once

using namespace std;

static vector<float> ReadImagesMNIST( string filePath );

static vector<float> ReadLabelsMNIST( string filePath );

static void MNIST()
{
    vector<float> trainImages = ReadImagesMNIST("../Showcase/MNIST/train-images-idx3-ubyte");
    vector<float> trainLabels = ReadLabelsMNIST("../Showcase/MNIST/train-labels-idx1-ubyte");
    vector<float> testImages = ReadImagesMNIST("../Showcase/MNIST/t10k-images-idx3-ubyte");
    vector<float> testLabels = ReadLabelsMNIST("../Showcase/MNIST/t10k-labels-idx1-ubyte");

    trainImages.resize( 784 * 100 );
    trainLabels.resize( 100 );
    testImages.resize( 784 * 100 );
    testLabels.resize( 100 );

    Flow::NArray xTrain = Flow::Create( { 100, 784 }, trainImages );
    Flow::NArray yTrain = Flow::Create( { 100 },      trainLabels );
    Flow::NArray xTest  = Flow::Create( { 100, 784 }, testImages );
    Flow::NArray yTest  = Flow::Create( { 100 },      trainLabels );

    for ( int i = 0; i < 28; i++ )
    {
        for ( int j = 0; j < 28; j++ )
            cout << setw(3) << right << xTrain.Get({ 73, i * 28 + j }) << " ";
        cout << endl;
    }
    cout << "Label: " << trainLabels[73] << endl;

    Flow::NArray w1 = Flow::Random({ 784, 128 });
    Flow::NArray b1 = Flow::Random({ 128 });
    Flow::NArray w2 = Flow::Random({ 128, 10 });
    Flow::NArray b2 = Flow::Random({ 10 });
    
    float learningRate = 0.1f;

    for ( int epoch = 0; epoch < 100; epoch++ )
    {
        Flow::NArray a = ReLU( Add( MM( xTrain, w1 ), b1 ) );
        Flow::NArray yPred = Add( MM( a, w2 ), b2 );
        Flow::NArray loss = Flow::CrossEntropy( yPred, yTrain );

        w1.GetGradient().Reset(0);
        b1.GetGradient().Reset(0);
        w2.GetGradient().Reset(0);
        b2.GetGradient().Reset(0);

        loss.Backpropagate();

        w1 = Sub( w1.Copy(), Mul( w1.GetGradient(), learningRate ) );
        b1 = Sub( b1.Copy(), Mul( b1.GetGradient(), learningRate ) );
        w2 = Sub( w2.Copy(), Mul( w2.GetGradient(), learningRate ) );
        b2 = Sub( b2.Copy(), Mul( b2.GetGradient(), learningRate ) );

        Flow::Print(loss);
    }
}

vector<float> ReadImagesMNIST( string filePath )
{
    vector< vector< unsigned char > > images;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        Flow::Print("[Error] Cannot open MNIST image file.");
    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if( magicNumber != 2051 )
        Flow::Print("[Error] Invalid MNIST image file.");
    file.read( (char*)&numImages, 4 );
    numImages = __builtin_bswap32(numImages);
    file.read( (char*)&rows, 4 );
    rows = __builtin_bswap32(rows);
    file.read( (char*)&cols, 4 );
    cols = __builtin_bswap32(cols);
    for ( int i = 0; i < numImages; i++ )
    {
        vector< unsigned char > image( rows * cols );
        file.read( (char*)image.data(), rows * cols );
        images.push_back(image);
    }
    file.close();
    vector<float> imagesData;
    for ( auto v : images )
    {
        for ( auto c : v )
            imagesData.push_back(static_cast<float>(c));
    }
    return imagesData;
}

vector<float> ReadLabelsMNIST( string filePath )
{
    vector< unsigned char > labels;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        Flow::Print("[Error] Cannot open MNIST label file.");
    int magicNumber = 0, numLabels = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if ( magicNumber != 2049 )
        Flow::Print("[Error] Invalid MNIST label file.");
    file.read( (char*)&numLabels, 4 );
    numLabels = __builtin_bswap32(numLabels);
    labels.resize(numLabels);
    file.read( (char*)labels.data(), numLabels );
    file.close();
    vector<float> labelsData;
    for ( auto c : labels )
        labelsData.push_back(static_cast<float>(c));
    return labelsData;
}