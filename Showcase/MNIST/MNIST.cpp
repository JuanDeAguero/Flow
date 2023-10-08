// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Flow.h"

using namespace std;

static vector<float> ReadImagesMNIST( string filePath );

static vector<float> ReadLabelsMNIST( string filePath );

int main()
{
    vector<float> trainImages = ReadImagesMNIST("../train-images-idx3-ubyte");
    vector<float> trainLabels = ReadLabelsMNIST("../train-labels-idx1-ubyte");
    vector<float> testImages = ReadImagesMNIST("../t10k-images-idx3-ubyte");
    vector<float> testLabels = ReadLabelsMNIST("../t10k-labels-idx1-ubyte");

    trainImages.resize( 784 * 100 );
    trainLabels.resize( 100 );
    testImages.resize( 784 * 100 );
    testLabels.resize( 100 );

    Flow::NArray xTrain = Flow::Create( { 100, 784 }, trainImages );
    Flow::NArray yTrain = Flow::Create( { 100 }, trainLabels );
    Flow::NArray xTest = Flow::Create( { 100, 784 }, testImages );
    Flow::NArray yTest = Flow::Create( { 100 }, trainLabels );

    Flow::Print("add started");
    Flow::NArray a = Add( xTrain.Copy(), xTrain.Copy() );
    Flow::Print("add finished");

    xTrain = Flow::Div( xTrain.Copy(), Flow::Create( { 1 }, { 255 } ) );
    xTest = Flow::Div( xTest.Copy(), Flow::Create( { 1 }, { 255 } ) );

    for ( int i = 0; i < 28; i++ )
    {
        for ( int j = 0; j < 28; j++ )
            cout << setw(3) << right << xTrain.Get({ 76, i * 28 + j }) << " ";
        cout << endl;
    }
    cout << "Label: " << trainLabels[76] << endl;

    Flow::NArray w1 = Flow::Random({ 784, 128 });
    Flow::NArray b1 = Flow::Random({ 128 });
    Flow::NArray w2 = Flow::Random({ 128, 10 });
    Flow::NArray b2 = Flow::Random({ 10 });
    
    float learningRate = 0.1f;

    for ( int epoch = 0; epoch < 100; epoch++ )
    {
        Flow::NArray a = ReLU( Add( MM( xTrain, w1 ), b1 ) );
        Flow::NArray yPred = Add( MM( a, w2 ), b2 );
        Flow::NArray loss = CrossEntropy( yPred, yTrain );

        w1.GetGradient().Reset(0);
        b1.GetGradient().Reset(0);
        w2.GetGradient().Reset(0);
        b2.GetGradient().Reset(0);

        loss.Backpropagate();

        w1 = Sub( w1.Copy(), Mul( w1.GetGradient().Copy(), learningRate ) );
        b1 = Sub( b1.Copy(), Mul( b1.GetGradient().Copy(), learningRate ) );
        w2 = Sub( w2.Copy(), Mul( w2.GetGradient().Copy(), learningRate ) );
        b2 = Sub( b2.Copy(), Mul( b2.GetGradient().Copy(), learningRate ) );

        Flow::Print(loss);
    }
}

vector<float> ReadImagesMNIST( string filePath )
{
    vector< vector< unsigned char > > images;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        throw runtime_error("[ReadImagesMNIST] Cannot open MNIST image file.");
    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if( magicNumber != 2051 )
        throw runtime_error("[ReadImagesMNIST] Invalid MNIST image file.");
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
    for ( auto image : images )
    {
        for ( auto imageChar : image )
            imagesData.push_back(static_cast<float>(imageChar));
    }
    return imagesData;
}

vector<float> ReadLabelsMNIST( string filePath )
{
    vector< unsigned char > labels;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        throw runtime_error("[ReadLabelsMNIST] Cannot open MNIST label file.");
    int magicNumber = 0, numLabels = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if ( magicNumber != 2049 )
        throw runtime_error("[ReadLabelsMNIST] Invalid MNIST label file.");
    file.read( (char*)&numLabels, 4 );
    numLabels = __builtin_bswap32(numLabels);
    labels.resize(numLabels);
    file.read( (char*)labels.data(), numLabels );
    file.close();
    vector<float> labelsData;
    for ( auto label : labels )
        labelsData.push_back(static_cast<float>(label));
    return labelsData;
}