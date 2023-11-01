// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "Flow.h"

using namespace std;

int main()
{
    ifstream file("../names.txt");
    vector<string> names;
    string line;
    //while ( getline( file, line ) ) names.push_back(line);
    getline( file, line ); names.push_back(line);
    file.close();
    //Flow::Print(names.size());

    vector<char> chars(27);
    chars[0] = '.';
    for ( int i = 0; i < 26; i++ ) chars[ i + 1 ] = 'a' + i;
    map< char, int > charToIndex;
    for ( int i = 0; i < chars.size(); i++ ) charToIndex[chars[i]] = i;

    vector<int> xs, ys;
    map< pair< char, char >, int > bigramFreq;
    for ( string& name : names )
    {
        string chars = "." + name + ".";
        for ( int i = 0; i < chars.size() - 1; i++ )
        {
            char char1 = chars[i];
            char char2 = chars[ i + 1 ];
            int index1 = charToIndex[char1];
            int index2 = charToIndex[char2];
            xs.push_back(index1);
            ys.push_back(index2);
        }
    }

    Flow::NArray xTrain = Flow::OneHot( xs, 27 );
    Flow::NArray yTrain = Flow::OneHot( ys, 27 );
    Flow::NArray w = Flow::Random({ 27, 27 });

    for ( int epoch = 0; epoch < 100; epoch++ )
    {
        Flow::NArray logits = MM( xTrain, w );
        Flow::NArray counts = Flow::Exp(logits);
        Flow::NArray probs = Flow::Div( counts, Sum( counts, 1 ) );
        Flow::NArray loss = Flow::Neg( Mean( Log( Gather( probs, 1, Unsqueeze( yTrain, 1 ) ) ) ) );
        w.GetGradient().Reset(0);
        loss.Backpropagate();
        w = Flow::Sub( w.Copy(), Mul( w.GetGradient().Copy(), 10.0f ) );
        float sum = 0.0f;
        for ( float value : loss.Get() ) sum += value;
        Flow::Print(sum);
    }
}