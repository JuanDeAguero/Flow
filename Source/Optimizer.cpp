// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/Optimizer.h"

Flow::Optimizer::Optimizer( vector<NARRAY> arrays, float learningRate, float epsilon,
    float weightDecay )
    : LearningRate(learningRate), Epsilon(epsilon), WeightDecay(weightDecay), Time(0)
{
    for ( NARRAY arr : arrays )
    {
        Arrays.push_back(arr);
        Beta1s.push_back( NArray::Create( { 1 }, { 0.9f } ) );
        Beta2s.push_back( NArray::Create( { 1 }, { 0.999f } ) );
        Ms.push_back( NArray::Create( { 1 }, { 0.0f } ) );
        Vs.push_back( NArray::Create( { 1 }, { 0.0f } ));
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
        /*Arrays[i]->Copy(
            Sub( Arrays[i]->Copy(), Mul( Arrays[i]->GetGradient()->Copy(), LearningRate ) ) );*/

        NARRAY gradient = Arrays[i]->GetGradient()->Copy();
        NARRAY one = NArray::Create( { 1 }, { 1.0f } );
        NARRAY m = Ms[i]->Copy();
        NARRAY v = Vs[i]->Copy();
        Ms[i] = Add(
            Mul( Beta1s[i]->Copy(), m ),
            Mul( Sub( one->Copy(), Beta1s[i]->Copy() ), gradient->Copy() )
        );
        Vs[i] = Add(
            Mul( Beta2s[i]->Copy(), v ),
            Mul( Sub( one->Copy(), Beta2s[i]->Copy() ), Pow( gradient->Copy(), 2.0f ) )
        );
        NARRAY mHat = Div(
            Ms[i]->Copy(),
            Sub( one->Copy(), Pow( Beta1s[i]->Copy(), (float)Time ) )
        );
        NARRAY vHat = Div(
            Vs[i]->Copy(),
            Sub( one->Copy(), Pow( Beta2s[i]->Copy(), (float)Time ) )
        );
        NARRAY epsilon = NArray::Create( { 1 }, { Epsilon } );
        NARRAY weightDecay = NArray::Create( { 1 }, { WeightDecay } );
        NARRAY a = Add(
            Div( mHat, Add( Pow( vHat, 0.5f ), epsilon ) ),
            Mul( weightDecay, Arrays[i]->Copy() )
        );
        Arrays[i]->Copy( Sub( Arrays[i]->Copy(), Mul( a->Copy(), LearningRate ) ) );
    }
}