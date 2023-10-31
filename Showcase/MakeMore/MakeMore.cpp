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
    while ( getline( file, line ) ) names.push_back(line);
    file.close();
    Flow::Print(names.size());

    vector<char> chars(27);
    chars[0] = '.';
    for ( int i = 0; i < 26; i++ ) chars[ i + 1 ] = 'a' + i;
    map< char, int > charToIndex;
    for ( int i = 0; i < chars.size(); i++ ) charToIndex[chars[i]] = i;

    Flow::NArray N = Flow::Zeros({ 27, 27 });
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
            N.Set( { index1, index2 }, N.Get({ index1, index2 }) + 1 );
        }
    }
    Flow::Print(N);

    vector<float> pData(27);
    for ( int i = 0; i < 27; i++ ) pData[i] = N.Get({ 0, i });
    Flow::NArray p = Flow::Create( { 27 }, pData );
    p = Flow::Div( p.Copy(), Flow::Sum( p.Copy(), 0 ) );
    Flow::Print(p);
}