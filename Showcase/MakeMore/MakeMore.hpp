// Copyright (c) 2023 Juan M. G. de Agüero

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "Flow.h"

#pragma once

using namespace std;

static void MakeMore()
{
    ifstream file("../Showcase/names.txt");
    vector<string> names;
    string line;
    while (getline( file, line ))
        names.push_back(line);
    file.close();
    Flow::Log(names.size());
    map< pair< char, char >, int > bigramFreq;
    for ( string& name : names )
    {
        string chars = ">" + name + "<";
        for ( int i = 0; i < chars.size() - 1; i++ )
        {
            char char1 = chars[i];
            char char2 = chars[ i + 1 ];
            pair< char, char > bigram = make_pair( char1, char2 );
            if ( bigramFreq.find(bigram) != bigramFreq.end() )
                bigramFreq[bigram]++;
            else bigramFreq[bigram] = 1;
        }
    }
    for ( auto& entry : bigramFreq )
    {
        string char1 = string( 1, entry.first.first );
        string char2 = string( 1, entry.first.second );
        Flow::Log( char1 + " " + char2 + " " + to_string(entry.second) );
    }
}