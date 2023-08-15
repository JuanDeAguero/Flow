// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/Log.h"
#include "Flow/NArray.h"

using namespace std;

static void SimpleNN()
{
    Flow::NArray* xs = new Flow::NArray( { 3, 2 },
    {
        2.0f, 9.0f,  // 2 hours studied and 9 hours sleeping
        1.0f, 5.0f,
        3.0f, 6.0f
    } );

    Flow::NArray* ys = new Flow::NArray( { 3, 1 },
    {
        92.0f,  // grade
        100.0f,
        89.0f
    } );

    Flow::NArray* xsNorm = new Flow::NArray( { 3, 2 }, { 0.6667, 1.0000, 0.3333, 0.5556, 1.0000, 0.6667 } );
    Flow::NArray* ysNorm = new Flow::NArray( { 3, 1 }, { 0.92f, 1.0f, 0.89f } );

    Flow::NArray* w1 = Flow::Random({ 2, 3 });
    Flow::NArray* b1 = Flow::Random({ 3 });
    Flow::NArray* w2 = Flow::Random({ 3, 1 });
    Flow::NArray* b2 = Flow::Random({ 1 });

    float learningRate = 0.1f;

    for ( int epoch = 0; epoch < 10000; epoch++ )
    {
        Flow::NArray* h = Add( ( MMult( xsNorm, w1 ) ), b1 );
        Flow::NArray* yPred = Tanh( Add( MMult( h, w2 ), b2 ) );

        Flow::NArray* loss = Pow( Sub( yPred, ysNorm ), 2.0f );

        w1->GetGradient()->Reset(0);
        b1->GetGradient()->Reset(0);
        w2->GetGradient()->Reset(0);
        b2->GetGradient()->Reset(0);

        loss->Backpropagate();

        w1 = Sub( w1, Mult( w1->GetGradient(), learningRate ) )->Copy();
        b1 = Sub( b1, Mult( b1->GetGradient(), learningRate ) )->Copy();
        w2 = Sub( w2, Mult( w2->GetGradient(), learningRate ) )->Copy();
        b2 = Sub( b2, Mult( b2->GetGradient(), learningRate ) )->Copy();

        Flow::Log( loss->Get()[0] + loss->Get()[2] + loss->Get()[2], 20 );
    }

    Flow::NArray* test = new Flow::NArray( { 2, 2 }, { 0.6667, 1.0000, 1.0000, 0.6667 } );
    Flow::NArray* hTest = Add( ( MMult( test, w1 ) ), b1 );
    Flow::NArray* yPredTest = Tanh( Add( MMult( hTest, w2 ), b2 ) );
    Flow::Log(yPredTest);
}