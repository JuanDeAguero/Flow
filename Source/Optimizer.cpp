// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/Optimizer.h"

Flow::Optimizer::Optimizer( vector<NARRAY> arrays, float learningRate, float epsilon,
    float weightDecay )
    : LearningRate(learningRate), Epsilon(epsilon), WeightDecay(weightDecay), Time(0)
{
    for ( NARRAY arr : arrays )
    {
        Arrays.push_back(arr);
        Beta1s.push_back( Create( { 1 }, { 0.9f } ) );
        Beta2s.push_back( Create( { 1 }, { 0.999f } ) );
        Ms.push_back( Create( { 1 }, { 0.0f } ) );
        Vs.push_back( Create( { 1 }, { 0.0f } ));
    }
}

void Flow::Optimizer::ZeroGrad()
{
    for ( NARRAY arr : Arrays ) arr->GetGradient()->Reset(0.0f);
}

void Flow::Optimizer::Step()
{
    Time++;
    for ( int i = 0; i < Arrays.size(); i++ )
    {
        NARRAY gradient = Arrays[i]->GetGradient();
        NARRAY one = Create( { 1 }, { 1.0f } );
        Ms[i] = Add( Mul( Beta1s[i], Ms[i]->Copy() ), Mul( Sub( one, Beta1s[i] ), gradient ) );
        Vs[i] = Add(
            Mul( Beta2s[i], Vs[i]->Copy() ),
            Mul( Sub( one, Beta2s[i] ), Pow( gradient, 2.0f ) ) );
        NARRAY mHat = Div( Ms[i], Sub( one, Pow( Beta1s[i], (float)Time ) ) );
        NARRAY vHat = Div( Vs[i], Sub( one, Pow( Beta2s[i], (float)Time ) ) );
        NARRAY epsilon = Create( { 1 }, { Epsilon } );
        NARRAY weightDecay = Create( { 1 }, { WeightDecay } );
        NARRAY a = Add(
            Div( mHat, Add( Pow( vHat, 0.5f ), epsilon ) ),
            Mul( weightDecay, Arrays[i] )
        );
        Arrays[i]->Copy( Sub( Arrays[i], Mul( a, LearningRate ) ) );
    }
}