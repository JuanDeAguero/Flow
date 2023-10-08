// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow.h"

int main()
{
    Flow::NArray x = Flow::Create( { 3, 2 },
    {
        2.0f, 9.0f,  // 2 hours studied and 9 hours sleeping
        1.0f, 5.0f,
        3.0f, 6.0f
    } );

    Flow::NArray y = Flow::Create( { 3, 1 },
    {
        92.0f,  // grade
        100.0f,
        89.0f
    } );

    Flow::NArray xNorm = Flow::Create( { 3, 2 }, { 0.6667, 1.0000, 0.3333, 0.5556, 1.0000, 0.6667 } );
    Flow::NArray yNorm = Flow::Create( { 3, 1 }, { 0.92f, 1.0f, 0.89f } );

    Flow::NArray w1 = Flow::Random({ 2, 3 });
    Flow::NArray b1 = Flow::Random({ 3 });
    Flow::NArray w2 = Flow::Random({ 3, 1 });
    Flow::NArray b2 = Flow::Random({ 1 });

    float learningRate = 0.1f;

    for ( int epoch = 0; epoch < 10000; epoch++ )
    {
        Flow::NArray a = Tanh( Add( MM( xNorm, w1 ), b1 ) );
        Flow::NArray yPred = Add( MM( a, w2 ), b2 );
        Flow::NArray loss = Pow( Sub( yPred, yNorm ), 2.0f );

        w1.GetGradient().Reset(0);
        b1.GetGradient().Reset(0);
        w2.GetGradient().Reset(0);
        b2.GetGradient().Reset(0);

        loss.Backpropagate();

        w1 = Sub( w1.Copy(), Mul( w1.GetGradient().Copy(), learningRate ) );
        b1 = Sub( b1.Copy(), Mul( b1.GetGradient().Copy(), learningRate ) );
        w2 = Sub( w2.Copy(), Mul( w2.GetGradient().Copy(), learningRate ) );
        b2 = Sub( b2.Copy(), Mul( b2.GetGradient().Copy(), learningRate ) );

        Flow::Print( loss.Get()[0] + loss.Get()[2] + loss.Get()[2], 20 );
    }

    Flow::NArray xTest = Flow::Create( { 2, 2 }, { 0.6667, 1.0000, 1.0000, 0.6667 } );
    Flow::NArray a = Tanh( Add( MM( xTest, w1 ), b1 ) );
    Flow::NArray yPred = Add( MM( a, w2 ), b2 );
    Flow::Print(yPred);
}