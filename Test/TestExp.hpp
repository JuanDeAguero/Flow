// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArray.h"
#include "Flow/Print.h"
#include "Test.hpp"

static bool Test_Exp()
{
    int numPassed = 0;
    int numTests = 5;
    Flow::NArray::Operation op = Flow::NArray::Operation::EXP;

    Test( 1, numPassed,
        Flow::NArray::Create( { 2, 4 }, { 5, 7, 8, 9, 3, 4, 5, 4 } ), nullptr, {}, {}, {}, {}, op,
        { 148.4132, 1096.6332, 2980.9580, 8103.0840, 20.0855, 54.5981, 148.4132, 54.5981 },
        { 2, 4 },
        { 148.4132, 1096.6332, 2980.9580, 8103.0840, 20.0855, 54.5981, 148.4132, 54.5981 },
        { 2, 4 }, {}, {} );

    Test( 2, numPassed,
        Flow::NArray::Create( { 3 }, { 1, 2, 3 } ), nullptr, {}, {}, {}, {}, op,
        { 2.7183, 7.3891, 20.0855 },
        { 3 },
        { 2.7183, 7.3891, 20.0855 },
        { 3 }, {}, {} );

    Test( 3, numPassed,
        Flow::NArray::Create( { 2, 2 }, { 0, -1, -2, -3 } ), nullptr, {}, {}, {}, {}, op,
        { 1.0, 0.3679, 0.1353, 0.0498 },
        { 2, 2 },
        { 1.0, 0.3679, 0.1353, 0.0498 },
        { 2, 2 }, {}, {} );

    Test( 4, numPassed,
        Flow::NArray::Create( { 1, 4 }, { -1, 0, 1, 2 } ), nullptr, {}, {}, {}, {}, op,
        { 0.3679, 1.0, 2.7183, 7.3891 },
        { 1, 4 },
        { 0.3679, 1.0, 2.7183, 7.3891 },
        { 1, 4 }, {}, {} );

    Test( 5, numPassed,
        Flow::NArray::Create( { 2, 2, 2 }, { 0, 1, -1, 2, 1.5, -1.5, 2.5, -2.5 } ), nullptr, {}, {}, {}, {}, op,
        { 1.0, 2.7183, 0.3679, 7.3891, 4.4817, 0.2231, 12.1825, 0.0821 },
        { 2, 2, 2 },
        { 1.0, 2.7183, 0.3679, 7.3891, 4.4817, 0.2231, 12.1825, 0.0821 },
        { 2, 2, 2 }, {}, {} );

    Flow::Print( "Test_Exp " + to_string(numPassed) + "/" + to_string(numTests) );
    if ( numPassed == numTests ) return true;
    else return false;
}