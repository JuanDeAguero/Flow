// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/Optimizer.h"

Flow::Optimizer::Optimizer( vector<reference_wrapper<NArray>> arrays, float learningRate, float epsilon, float weightDecay )
    : LearningRate(learningRate), Epsilon(epsilon), WeightDecay(weightDecay), Time(0)
{
    for ( NArray& arr : arrays )
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
    for ( NArray& arr : Arrays ) arr.ResetGradient(0.0f);
}

void Flow::Optimizer::Step()
{
    Time++;
    for ( int i = 0; i < Arrays.size(); i++ )
    {
        //Arrays[i].get().Assign( Sub( Arrays[i].get().Copy(), Mul( Arrays[i].get().CopyGradient(), LearningRate ) ) );
        //continue;
        NArrayCore* gradient = Arrays[i].get().GetCore()->GetGradient()->Copy();
        NArrayCore* one = new NArrayCore( { 1 }, { 1.0f } );
        NArrayCore* m = Ms[i]->Copy();
        NArrayCore* v = Vs[i]->Copy();
        Ms[i]->Destroy();
        Vs[i]->Destroy();
        Ms[i] = Add( Mul( Beta1s[i]->Copy(), m ), Mul( Sub( one->Copy(), Beta1s[i]->Copy() ), gradient->Copy() ) );
        Vs[i] = Add( Mul( Beta2s[i]->Copy(), v ), Mul( Sub( one->Copy(), Beta2s[i]->Copy() ), Pow( gradient->Copy(), 2.0f ) ) );
        NArrayCore* mHat = Div( Ms[i]->Copy(), Sub( one->Copy(), Pow( Beta1s[i]->Copy(), (float)Time ) ) );
        NArrayCore* vHat = Div( Vs[i]->Copy(), Sub( one->Copy(), Pow( Beta2s[i]->Copy(), (float)Time ) ) );
        NArrayCore* epsilon = new NArrayCore( { 1 }, { Epsilon } );
        NArrayCore* weightDecay = new NArrayCore( { 1 }, { WeightDecay } );
        NArrayCore* a = Add( Div( mHat, Add( Pow( vHat, 0.5f ), epsilon ) ), Mul( weightDecay, Arrays[i].get().Copy() ) );
        Arrays[i].get().Assign( Sub( Arrays[i].get().Copy(), Mul( a->Copy(), LearningRate ) ) );
        a->Destroy();
        gradient->Destroy();
    }
}