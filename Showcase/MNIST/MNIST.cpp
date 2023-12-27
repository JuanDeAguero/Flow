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

static vector<int> ReadLabelsMNIST( string filePath );

int main()
{
    vector<float> trainImages = ReadImagesMNIST("../train-images-idx3-ubyte");
    vector<float> testImages = ReadImagesMNIST("../t10k-images-idx3-ubyte");
    vector<int> trainLabels = ReadLabelsMNIST("../train-labels-idx1-ubyte");
    vector<int> testLabels = ReadLabelsMNIST("../t10k-labels-idx1-ubyte");

    int n = 6000;

    trainImages.resize( 784 * n );
    testImages.resize( 784 * n );
    trainLabels.resize( n );
    testLabels.resize( n );

    Flow::NArray xTrain( Flow::Create( { n, 784 }, trainImages ) );
    Flow::NArray xTest( Flow::Create( { n, 784 }, testImages ) );
    Flow::NArray yTrain( Flow::OneHot( trainLabels, 10 ) );
    Flow::NArray yTest( Flow::OneHot( testLabels, 10 ) );

    xTrain.Assign( Flow::Div( xTrain.Copy(), Flow::Create( { 1 }, { 255.0f } ) ) );
    xTest.Assign( Flow::Div( xTest.Copy(), Flow::Create( { 1 }, { 255.0f } ) ) );

    Flow::NArray w1( Flow::Random({ 784, 512 }) );
    Flow::NArray b1( Flow::Random({ 512 }) );
    Flow::NArray w2( Flow::Random({ 512, 256 }) );
    Flow::NArray b2( Flow::Random({ 256 }) );
    Flow::NArray w3( Flow::Random({ 256, 10 }) );
    Flow::NArray b3( Flow::Random({ 10 }) );
    
    float learningRate = 0.9f;
    int maxEpochs = 100;

    for ( int epoch = 0; epoch < maxEpochs; epoch++ )
    {
        Flow::NArray a1( ReLU( Add( MM( xTrain, w1 ), b1 ) ) );
        Flow::NArray a2( ReLU( Add( MM( a1, w2 ), b2 ) ) );
        Flow::NArray yPred( Add( MM( a2, w3 ), b3 ) );
        Flow::NArray loss( CrossEntropy( yPred, yTrain ) );

        w1.GetGradient()->Reset(0);
        b1.GetGradient()->Reset(0);
        w2.GetGradient()->Reset(0);
        b2.GetGradient()->Reset(0);
        w3.GetGradient()->Reset(0);
        b3.GetGradient()->Reset(0);

        loss.Backpropagate();

        w1.Assign( Flow::Sub( w1.Copy(), Mul( w1.GetGradient()->Copy(), learningRate ) ) );
        b1.Assign( Flow::Sub( b1.Copy(), Mul( b1.GetGradient()->Copy(), learningRate ) ) );
        w2.Assign( Flow::Sub( w2.Copy(), Mul( w2.GetGradient()->Copy(), learningRate ) ) );
        b2.Assign( Flow::Sub( b2.Copy(), Mul( b2.GetGradient()->Copy(), learningRate ) ) );
        w3.Assign( Flow::Sub( w3.Copy(), Mul( w3.GetGradient()->Copy(), learningRate ) ) );
        b3.Assign( Flow::Sub( b3.Copy(), Mul( b3.GetGradient()->Copy(), learningRate ) ) );

        Flow::Print( "epoch: " + to_string( epoch + 1 ) + "  loss: " + to_string(loss.Get()[0]) + "  free: " + to_string(Flow::GetCUDAFreeMemory()) );
    }
    
    n = 100;
    int numCorrect = 0;
    for ( int i = 0; i < n; i++ )
    {
        vector<float> testData;
        for ( int j = 0; j < 28; j++ )
        {
            for ( int k = 0; k < 28; k++ )
                testData.push_back( xTest.Get({ i, j * 28 + k }) );
        }
        Flow::NArray test( Flow::Create( { 1, 784 }, testData ) );
        Flow::NArray a1( ReLU( Add( MM( test, w1 ), b1 ) ) );
        Flow::NArray a2( ReLU( Add( MM( a1, w2 ), b2 ) ) );
        Flow::NArray yPred( Add( MM( a2, w3 ), b3 ) );
        float maxVal = yPred.Get()[0];
        int maxIndex = 0;
        for ( int j = 1; j < 10; j++ )
        {
            if ( yPred.Get()[j] > maxVal )
            {
                maxVal = yPred.Get()[j];
                maxIndex = j;
            }
        }
        if ( testLabels[i] == maxIndex ) numCorrect++;
    }
    float accuracy = (float)(numCorrect) / (float)(n) * 100.0f;
    Flow::Print( to_string(accuracy) + "%" );
}

vector<float> ReadImagesMNIST( string filePath )
{
    vector< vector< unsigned char > > images;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        throw runtime_error("Cannot open MNIST image file.");
    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = _byteswap_ulong(magicNumber);
    if( magicNumber != 2051 )
        throw runtime_error("Invalid MNIST image file.");
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
        for ( auto imageChar : image )
            imagesData.push_back((float)imageChar);
    }
    return imagesData;
}

vector<int> ReadLabelsMNIST( string filePath )
{
    vector< unsigned char > labels;
    ifstream file( filePath, ios::binary );
    if (!file.is_open())
        throw runtime_error("Cannot open MNIST label file.");
    int magicNumber = 0, numLabels = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = _byteswap_ulong(magicNumber);
    if ( magicNumber != 2049 )
        throw runtime_error("Invalid MNIST label file.");
    file.read( (char*)&numLabels, 4 );
    numLabels = _byteswap_ulong(numLabels);
    labels.resize(numLabels);
    file.read( (char*)labels.data(), numLabels );
    file.close();
    vector<int> labelsData;
    for ( auto label : labels )
        labelsData.push_back((int)label);
    return labelsData;
}