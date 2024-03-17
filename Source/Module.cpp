// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/Module.h"

#include "Flow/Print.h"

using namespace std;

Flow::Module::Module()
{

}

NARRAY Flow::Module::Forward( NARRAY arr )
{
    return arr;
}

vector<NARRAY> Flow::Module::GetParameters()
{
    vector<NARRAY> params;
    for ( auto& module : Modules )
    {
        auto moduleParams = module->GetParameters();
        params.insert( params.end(), moduleParams.begin(), moduleParams.end() );
    }
    return params;
}

Flow::Linear::Linear() {}

Flow::Linear::Linear( vector<int> weightShape, vector<int> biasShape )
{
    Weight = Random(weightShape);
    Bias = Random(biasShape);
}

shared_ptr<Flow::Linear> Flow::Linear::Create( vector<int> weightShape, vector<int> biasShape )
{
    return make_shared<Linear>( weightShape, biasShape );
}

NARRAY Flow::Linear::Forward( NARRAY arr )
{
    return Add( Matmul( arr, Weight ), Bias );
}

vector<NARRAY> Flow::Linear::GetParameters()
{
    return { Weight, Bias };
}