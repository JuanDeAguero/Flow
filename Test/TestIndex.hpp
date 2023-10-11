// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"

using namespace std;

static bool Test_Index()
{
    int numPassed = 0;

    Flow::NArray arr = Flow::Create( { 3, 4 }, { 0.1427, 0.0231, -0.5414, -1.0009, -0.4664, 0.2647, -0.1228, -1.1068, -1.1734, -0.6571, 0.7230, -0.6004 } );
    Flow::NArray index = Flow::Create( { 2 }, { 0, 2 } );
    Flow::NArray result = Flow::Index( arr, 0, index );
    std::vector<float> expectedData = { 0.1427,  0.0231, -0.5414, -1.0009, -1.1734, -0.6571, 0.7230, -0.6004 };
    vector<int> expectedShape = { 2, 4 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Index_1 PASSED"); numPassed++; }
    else Flow::Print("Test_Index_1 FAILED");

    index = Flow::Create( { 2 }, { 0, 2 } );
    result = Flow::Index( arr, 1, index );
    expectedData = { 0.1427, -0.5414, -0.4664, -0.1228, -1.1734,  0.7230 };
    expectedShape = { 3, 2 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Index_2 PASSED"); numPassed++; }
    else Flow::Print("Test_Index_2 FAILED");

    index = Flow::Create( { 3 }, { 1, 2, 3 } );
    result = Flow::Index( arr, 1, index );
    expectedData = { 0.0231, -0.5414, -1.0009, 0.2647, -0.1228, -1.1068, -0.6571,  0.7230, -0.6004 };
    expectedShape = { 3, 3 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Index_3 PASSED"); numPassed++; }
    else Flow::Print("Test_Index_3 FAILED");

    arr = Flow::Create( { 2, 3, 3, 4 }, { -1.1263, -0.1959, 0.8853, -0.3991, 0.6080, -1.1640, -1.7273, 2.4827, -0.2898, -0.5410, 0.2972, -0.6018, 1.5948, -0.1874, -0.6735, -0.2473, 1.5773, -0.3341, 1.1488, -0.9923, 0.4772, 0.1063, -0.4745, -1.3895, 0.0617, 1.1756, 0.9995, 0.5390, 0.3146, 0.3292, 1.4136, 1.0530, -1.0351, 0.8877, 0.1307, -0.8012, -0.0599, 0.3944, -1.3896, 0.9286, -1.6658, -0.4240, 0.5625, -0.5815, -1.7566, -0.2888, 1.1982, -0.2030, 0.2221, -0.2423, 1.5323, 0.0579, -0.0664, 0.4469, -0.9344, -1.6313, -0.4817, -0.7609, 0.7713, 0.3644, -0.8192, -0.3644, -1.2139, -0.4286, 1.9206, 0.0429, -0.5442, -0.0687, -1.1096, 0.0858, 0.4264, -2.134 } );
    index = Flow::Create( { 1 }, { 1 } );
    result = Flow::Index( arr, 3, index );
    expectedData = { -0.1959, -1.1640, -0.5410, -0.1874, -0.3341, 0.1063, 1.1756, 0.3292, 0.8877, 0.3944, -0.4240, -0.2888, -0.2423, 0.4469, -0.7609, -0.3644, 0.0429, 0.0858 };
    expectedShape = { 2, 3, 3, 1 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Index_4 PASSED"); numPassed++; }
    else Flow::Print("Test_Index_4 FAILED");

    arr = Flow::Create( { 2, 3, 3, 4 }, { -1.1497, 0.1809, 0.8588, 1.4287, -0.3405, -0.6588, 0.2813, -0.0682, -1.0958, 0.5181, -0.6085, 2.2340, -1.5476, 0.0718, 0.3237, -1.2872, 0.7592, -0.0410, 1.6078, 0.2303, 0.0208, 0.0767, 1.6700, -1.9479, -0.5470, -1.0043, 0.3499, -0.2809, -0.3340, 0.2573, 1.9164, -0.0876, -0.3625, -0.3958, -0.1697, -1.3937, 1.4505, 1.2828, 0.1550, -0.2190, 0.5575, -0.2396, -0.2668, -0.5591, -2.1385, 2.1140, -2.3357, 0.1918, -0.4362, 0.0274, 0.5932, 0.8321, -0.7410, 0.7611, 0.6222, -1.1216, -0.3099, -1.4480, 2.0349, 0.3710, 0.6520, 0.3942, -0.2357, -0.6643, -0.0842, 1.9571, -0.7171, -1.0089, -0.0054, -0.1637, -0.1683, -0.5469 } );
    index = Flow::Create( { 2 }, { 1, 2 } );
    result = Flow::Index( arr, 2, index );
    expectedData = { -0.3405, -0.6588, 0.2813, -0.0682, -1.0958, 0.5181, -0.6085, 2.2340, 0.7592, -0.0410, 1.6078, 0.2303, 0.0208, 0.0767, 1.6700, -1.9479, -0.3340, 0.2573, 1.9164, -0.0876, -0.3625, -0.3958, -0.1697, -1.3937, 0.5575, -0.2396, -0.2668, -0.5591, -2.1385, 2.1140, -2.3357, 0.1918, -0.7410, 0.7611, 0.6222, -1.1216, -0.3099, -1.4480, 2.0349, 0.3710, -0.0842, 1.9571, -0.7171, -1.0089, -0.0054, -0.1637, -0.1683, -0.5469 };
    expectedShape = { 2, 3, 2, 4 };
    if ( expectedData == result.Get() && expectedShape == result.GetShape() ) { Flow::Print("Test_Index_5 PASSED"); numPassed++; }
    else Flow::Print("Test_Index_5 FAILED");

    int numTests = 5;
    Flow::Print( "Test_Index " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}