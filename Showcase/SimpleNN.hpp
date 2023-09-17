// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow.h"

#pragma once

static void SimpleNN()
{
    Flow::NArray xs = Flow::Create( { 3, 2 },
    {
        2.0f, 9.0f,  // 2 hours studied and 9 hours sleeping
        1.0f, 5.0f,
        3.0f, 6.0f
    } );

    Flow::NArray ys = Flow::Create( { 3, 1 },
    {
        92.0f,  // grade
        100.0f,
        89.0f
    } );

    Flow::NArray xsNorm = Flow::Create( { 3, 2 }, { 0.6667, 1.0000, 0.3333, 0.5556, 1.0000, 0.6667 } );
    Flow::NArray ysNorm = Flow::Create( { 3, 1 }, { 0.92f, 1.0f, 0.89f } );

    Flow::NArray w1 = Flow::Random({ 2, 3 });
    Flow::NArray b1 = Flow::Random({ 3 });
    Flow::NArray w2 = Flow::Random({ 3, 1 });
    Flow::NArray b2 = Flow::Random({ 1 });

    float learningRate = 0.1f;

    for ( int epoch = 0; epoch < 10000; epoch++ )
    {
        Flow::NArray h = Add( ( MMult( xsNorm, w1 ) ), b1 );
        Flow::NArray yPred = Tanh( Add( MMult( h, w2 ), b2 ) );

        Flow::NArray loss = Pow( Sub( yPred, ysNorm ), 2.0f );

        w1.GetGradient().Reset(0);
        b1.GetGradient().Reset(0);
        w2.GetGradient().Reset(0);
        b2.GetGradient().Reset(0);

        loss.Backpropagate();

        w1 = Sub( w1.Copy(), Mult( w1.GetGradient(), learningRate ) );
        b1 = Sub( b1.Copy(), Mult( b1.GetGradient(), learningRate ) );
        w2 = Sub( w2.Copy(), Mult( w2.GetGradient(), learningRate ) );
        b2 = Sub( b2.Copy(), Mult( b2.GetGradient(), learningRate ) );

        Flow::Log( loss.Get()[0] + loss.Get()[2] + loss.Get()[2], 20 );
    }

    Flow::NArray test = Flow::Create( { 2, 2 }, { 0.6667, 1.0000, 1.0000, 0.6667 } );
    Flow::NArray hTest = Add( ( MMult( test, w1 ) ), b1 );
    Flow::NArray yPredTest = Tanh( Add( MMult( hTest, w2 ), b2 ) );
    Flow::Log(yPredTest);
}