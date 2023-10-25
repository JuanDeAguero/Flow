// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Exp( NArrayCore* arr )
    {
        if (UseCUDA)
            return Exp_CUDA(arr);

        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = exp(value);
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::EXP );
    }
}

void Flow::NArrayCore::BackwardExp()
{
    if (UseCUDA)
    {
        BackwardExp_CUDA();
        return;
    }
    
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] * exp( Operands[0]->Data[i] );
        Operands[0]->Gradient->Data[i] += grad;
    }
}