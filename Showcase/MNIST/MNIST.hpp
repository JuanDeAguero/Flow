// Copyright (c) 2023 Juan M. G. de Agüero

#include <fstream>
#include <iostream>
#include <vector>

#include "Flow.h"

#pragma once

using namespace std;

vector<vector< unsigned char >> ReadImagesMNIST( string filename )
{
    vector<vector< unsigned char >> images;
    ifstream file( filename, ios::binary );
    if (!file.is_open())
        Flow::Log("[Error] Cannot open MNIST image file.");
    int magicNumber = 0, numOfImages = 0, rows = 0, cols = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if( magicNumber != 2051 )
        Flow::Log("[Error] Invalid MNIST image file.");
    file.read( (char*)&numOfImages, 4 );
    numOfImages = __builtin_bswap32(numOfImages);
    file.read( (char*)&rows, 4 );
    rows = __builtin_bswap32(rows);
    file.read( (char*)&cols, 4 );
    cols = __builtin_bswap32(cols);
    for ( int i = 0; i < numOfImages; i++ )
    {
        vector< unsigned char > image( rows * cols );
        file.read( (char*)image.data(), rows * cols );
        images.push_back(image);
    }
    file.close();
    return images;
}

vector<unsigned char> ReadLabelsMNIST( string filename )
{
    vector< unsigned char > labels;
    ifstream file( filename, ios::binary );
    if (!file.is_open())
        Flow::Log("[Error] Cannot open MNIST label file.");
    int magicNumber = 0, numOfLabels = 0;
    file.read( (char*)&magicNumber, 4 );
    magicNumber = __builtin_bswap32(magicNumber);
    if ( magicNumber != 2049 )
        Flow::Log("[Error] Invalid MNIST label file.");
    file.read( (char*)&numOfLabels, 4 );
    numOfLabels = __builtin_bswap32(numOfLabels);
    labels.resize(numOfLabels);
    file.read( (char*)labels.data(), numOfLabels );
    file.close();
    return labels;
}

static void MNIST()
{
    vector<vector< unsigned char >> images = ReadImagesMNIST("../Showcase/MNIST/train-images-idx3-ubyte");
    vector< unsigned char > labels = ReadLabelsMNIST("../Showcase/MNIST/train-labels-idx1-ubyte");
    for ( int i = 0; i < 28; i++ )
    {
        for ( int j = 0; j < 28; j++ )
            cout << setw(3) << right << static_cast<int>(images[0][ i * 28 + j ]) << " ";
        cout << endl;
    }
    cout << "Label: " << static_cast<int>(labels[0]) << endl;
}