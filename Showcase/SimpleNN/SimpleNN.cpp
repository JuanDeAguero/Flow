// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow.h"

int main()
{
    NARRAY x( Flow::NArray::Create( { 3, 2 },
    {
        2.0f, 9.0f,  // 2 hours studied and 9 hours sleeping.
        1.0f, 5.0f,
        3.0f, 6.0f
    } ) );

    NARRAY y( Flow::NArray::Create( { 3, 1 },
    {
        92.0f,  // Grade.
        100.0f,
        89.0f
    } ) );

    NARRAY xNorm( Flow::NArray::Create( { 3, 2 },
    {
        0.6667f, 1.0f,
        0.3333f, 0.5556f,
        1.0f, 0.6667f
    } ) );

    NARRAY yNorm( Flow::NArray::Create( { 3, 1 }, { 0.92f, 1.0f, 0.89f } ) );

    NARRAY w1( Flow::Random({ 2, 3 }) );
    NARRAY b1( Flow::Random({ 3 }) );
    NARRAY w2( Flow::Random({ 3, 1 }) );
    NARRAY b2( Flow::Random({ 1 }) );

    float learningRate = 0.1f;
    int maxEpochs = 100;

    for ( int epoch = 0; epoch < maxEpochs; epoch++ )
    {
        NARRAY a = Tanh( Add( MM( xNorm, w1 ), b1 ) );
        NARRAY yPredicted = Add( MM( a, w2 ), b2 );
        NARRAY loss = Pow( Sub( yPredicted, yNorm ), 2.0f );

        w1->GetGradient()->Reset(0.0f);
        b1->GetGradient()->Reset(0.0f);
        w2->GetGradient()->Reset(0.0f);
        b2->GetGradient()->Reset(0.0f);

        loss->Backpropagate();

        w1->Copy( Sub( w1->Copy(), Mul( w1->GetGradient()->Copy(), learningRate ) ) );
        b1->Copy( Sub( b1->Copy(), Mul( b1->GetGradient()->Copy(), learningRate ) ) );
        w2->Copy( Sub( w2->Copy(), Mul( w2->GetGradient()->Copy(), learningRate ) ) );
        b2->Copy( Sub( b2->Copy(), Mul( b2->GetGradient()->Copy(), learningRate ) ) );

        Flow::Print( loss->Get()[0] + loss->Get()[2] + loss->Get()[2], 20 );
    }

    NARRAY test = Flow::NArray::Create( { 2, 2 }, { 0.6667f, 1.0f, 1.0f, 0.6667f } );
    NARRAY a = Tanh( Add( MM( test, w1 ), b1 ) );
    NARRAY yPredicted = Add( MM( a, w2 ), b2 );
    Flow::Print(yPredicted);
}