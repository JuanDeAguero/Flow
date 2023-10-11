// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <random>
#include <vector>

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

using namespace std;

static bool Test_Add()
{
    int numPassed = 0;

    vector<float> d( 10000000, 123 );
    Flow::NArray a = Flow::Create( { 10000000 }, d );
    Flow::NArray b = Flow::Create( { 10000000 }, d );
    Flow::NArray c = Flow::Add( a, b );

    Test( 1, numPassed,
        Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ),
        Flow::Create( { 3 }, { 1, 10, 100 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 1, 11, 102, 4, 14, 105, 7, 17, 108 },
        { 3, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 3 },
        { 3, 3, 3 },
        { 3 } );

    Test( 2, numPassed,
        Flow::Create( { 3, 1 }, { 0, 1, 2 } ),
        Flow::Create( { 1, 3 }, { 3, 4, 5 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 3, 4, 5, 4, 5, 6, 5, 6, 7 },
        { 3, 3 },
        { 3, 3, 3 },
        { 3, 1 },
        { 3, 3, 3 },
        { 1, 3 } );

    Test( 3, numPassed,
        Flow::Create( { 2, 2 }, { 4, 1, 2, 3 } ),
        Flow::Create( { 2, 2 }, { 3, 6, 2, 5 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 7, 7, 4, 8 },
        { 2, 2 },
        { 1, 1, 1, 1 },
        { 2, 2 },
        { 1, 1, 1, 1 },
        { 2, 2 } );

    Test( 4, numPassed,
        Flow::Create( { 3 }, { 1, 2, 3 } ),
        Flow::Create( { 3 }, { 4, 5, 6 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 5, 7, 9 },
        { 3 },
        { 1, 1, 1 },
        { 3 },
        { 1, 1, 1 },
        { 3 } );

    Test( 5, numPassed,
        Flow::Create( { 3, 2 }, { 1, 2, 3, 4, 5, 6 } ),
        Flow::Create( { 2 }, { 1, 2 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 2, 4, 4, 6, 6, 8 },
        { 3, 2 },
        { 1, 1, 1, 1, 1, 1 },
        { 3, 2 },
        { 3, 3 },
        { 2 } );

    Test( 6, numPassed,
        Flow::Create( { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 } ),
        Flow::Create( { 1 }, { 10 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 11, 12, 13, 14, 15, 16, 17, 18, 19 },
        { 3, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 3 },
        { 9 },
        { 1 } );

    Test( 7, numPassed,
        Flow::Create( { 2 }, { 1, 2 } ),
        Flow::Create( { 3, 2 }, { 3, 4, 5, 6, 7, 8 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 4, 6, 6, 8, 8, 10 },
        { 3, 2 },
        { 3, 3 },
        { 2 },
        { 1, 1, 1, 1, 1, 1 },
        { 3, 2 } );

    Test( 8, numPassed,
        Flow::Create( { 2, 1 }, { 1, 2 } ),
        Flow::Create( { 2, 3 }, { 3, 4, 5, 6, 7, 8 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 4, 5, 6, 8, 9, 10 },
        { 2, 3 },
        { 3, 3 },
        { 2, 1 },
        { 1, 1, 1, 1, 1, 1 },
        { 2, 3 } );

    Test( 9, numPassed,
        Flow::Create( { 2 }, { 1, 2 } ),
        Flow::Create( { 2 }, { 3, 4 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 4, 6 },
        { 2 },
        { 1, 1 },
        { 2 },
        { 1, 1 },
        { 2 } );

    Test( 10, numPassed,
        Flow::Create( { 3, 1 }, { 1, 2, 3 } ),
        Flow::Create( { 1, 4 }, { 1, 2, 3, 4 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 },
        { 3, 4 },
        { 4, 4, 4 },
        { 3, 1 },
        { 3, 3, 3, 3 },
        { 1, 4 } );

    Test( 11, numPassed,
        Flow::Create( { 2, 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8,  8, 7, 6, 5, 4, 3, 2, 1, 0 } ),
        Flow::Create( { 3 }, { 1, 10, 100 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 1, 11, 102, 4, 14, 105, 7, 17, 108, 9, 17, 106, 6, 14, 103, 3, 11, 100 },
        { 2, 3, 3 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 2, 3, 3 },
        { 6, 6, 6 },
        { 3 } );

    Test( 12, numPassed,
        Flow::Create( { 2, 3, 1 }, { 0, 1, 2, 3, 4, 5 } ),
        Flow::Create( { 1, 1, 3 }, { 3, 4, 5 } ), {}, {},
        Flow::NArrayCore::Operation::ADD,
        { 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10 },
        { 2, 3, 3 },
        { 3, 3, 3, 3, 3, 3 },
        { 2, 3, 1 },
        { 6, 6, 6 },
        { 1, 1, 3 } );

    Flow::NArray arr1;
    Flow::NArray arr2;
    Flow::NArray result;
    vector<float> data;
    vector<int> shape;

    arr1 = Flow::Create( { 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 } );
    arr2 = Flow::Create( { 2, 2, 2 }, { 8, 7, 6, 5, 4, 3, 2, 1 } );
    result = Flow::Add( arr1, arr2 );
    data = { 9, 9, 9, 9, 9, 9, 9, 9 };
    shape = { 2, 2, 2 };
    if ( data == result.Get() && shape == result.GetShape() )  { Flow::Print("Test_Add_13 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_13 FAILED");

    arr1 = Flow::Create( { 2, 3, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 } );
    arr2 = Flow::Create( { 2 }, { 1, 2 } );
    result = Flow::Add( arr1, arr2 );
    data = { 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14 };
    shape = { 2, 3, 2 };
    if ( data == result.Get() && shape == result.GetShape() )  { Flow::Print("Test_Add_14 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_14 FAILED");

    arr1 = Flow::Create( { 2, 2, 2, 2 }, { 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16 } );
    arr2 = Flow::Create( { 2, 2, 2, 2 }, { 16, 14, 12, 10, 8, 6, 4, 2, 15, 13, 11, 9, 7, 5, 3, 1 } );
    result = Flow::Add( arr1, arr2 );
    data = { 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17 };
    shape = { 2, 2, 2, 2 };
    if ( data == result.Get() && shape == result.GetShape() )  { Flow::Print("Test_Add_15 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_15 FAILED");

    arr1 = Flow::Create( { 2, 2, 2, 2 }, { 11, 13, 15, 17, 12, 14, 16, 18, 13, 15, 17, 19, 14, 16, 18, 20 } );
    arr2 = Flow::Create( { 2 }, { 1, 3 } );
    result = Flow::Add( arr1, arr2 );
    data = { 12, 16, 16, 20, 13, 17, 17, 21, 14, 18, 18, 22, 15, 19, 19, 23 };
    shape = { 2, 2, 2, 2 };
    if ( data == result.Get() && shape == result.GetShape() )  { Flow::Print("Test_Add_16 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_16 FAILED");

    arr1 = Flow::Create( { 5, 2, 3, 2 }, { 51, 53, 52, 54, 53, 55, 54, 56, 55, 57, 56, 58, 57, 59, 58, 60, 59, 61, 60, 62, 61, 63, 62, 64, 63, 65, 64, 66, 65, 67, 66, 68, 67, 69, 68, 70, 69, 71, 70, 72, 71, 73, 72, 74, 73, 75, 74, 76, 75, 77, 76, 78, 77, 79, 78, 80, 79, 81, 80, 82 } );
    arr2 = Flow::Create( { 2, 3, 1 }, { 1, 2, 3, 4, 5, 6 } );
    result = Flow::Add( arr1, arr2 );
    data = { 52, 54, 54, 56, 56, 58, 58, 60, 60, 62, 62, 64, 58, 60, 60, 62, 62, 64, 64, 66, 66, 68, 68, 70, 64, 66, 66, 68, 68, 70, 70, 72, 72, 74, 74, 76, 70, 72, 72, 74, 74, 76, 76, 78, 78, 80, 80, 82, 76, 78, 78, 80, 80, 82, 82, 84, 84, 86, 86, 88 };
    shape = { 5, 2, 3, 2 };
    if ( data == result.Get() && shape == result.GetShape() )  { Flow::Print("Test_Add_17 PASSED"); numPassed++; }
    else Flow::Print("Test_Add_17 FAILED");

    int numTests = 17;
    Flow::Print( "Test_Add " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
}