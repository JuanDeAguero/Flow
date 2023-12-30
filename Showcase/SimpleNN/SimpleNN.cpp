// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow.h"

int main()
{
    Flow::NArray x( Flow::Create( { 3, 2 },
    {
        2.0f, 9.0f,  // 2 hours studied and 9 hours sleeping
        1.0f, 5.0f,
        3.0f, 6.0f
    } ) );

    Flow::NArray y( Flow::Create( { 3, 1 },
    {
        92.0f,  // grade
        100.0f,
        89.0f
    } ) );

    Flow::NArray xNorm( Flow::Create( { 3, 2 }, { 0.6667f, 1.0f, 0.3333f, 0.5556f, 1.0f, 0.6667f } ) );
    Flow::NArray yNorm( Flow::Create( { 3, 1 }, { 0.92f, 1.0f, 0.89f } ) );

    Flow::NArray w1( Flow::Random({ 2, 3 }) );
    Flow::NArray b1( Flow::Random({ 3 }) );
    Flow::NArray w2( Flow::Random({ 3, 1 }) );
    Flow::NArray b2( Flow::Random({ 1 }) );

    float learningRate = 0.1f;
    int maxEpochs = 1000;

    for ( int epoch = 0; epoch < maxEpochs; epoch++ )
    {
        Flow::NArray a( Tanh( Add( MM( xNorm, w1 ), b1 ) ) );
        Flow::NArray yPredicted( Add( MM( a, w2 ), b2 ) );
        Flow::NArray loss( Pow( Sub( yPredicted, yNorm ), 2.0f ) );

        w1.ResetGradient(0.0f);
        b1.ResetGradient(0.0f);
        w2.ResetGradient(0.0f);
        b2.ResetGradient(0.0f);

        loss.Backpropagate();

        w1.Assign( Sub( w1.Copy(), Mul( w1.CopyGradient(), learningRate ) ) );
        b1.Assign( Sub( b1.Copy(), Mul( b1.CopyGradient(), learningRate ) ) );
        w2.Assign( Sub( w2.Copy(), Mul( w2.CopyGradient(), learningRate ) ) );
        b2.Assign( Sub( b2.Copy(), Mul( b2.CopyGradient(), learningRate ) ) );

        Flow::Print( loss.Get()[0] + loss.Get()[2] + loss.Get()[2], 20 );
    }

    Flow::NArray test( Flow::Create( { 2, 2 }, { 0.6667f, 1.0f, 1.0f, 0.6667f } ) );
    Flow::NArray a( Tanh( Add( MM( test, w1 ), b1 ) ) );
    Flow::NArray yPredicted( Add( MM( a, w2 ), b2 ) );
    Flow::Print(yPredicted);
}