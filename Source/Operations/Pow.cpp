// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Pow( NArrayCore* arr, float exponent )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = pow( value, exponent );
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::POW );
        result->Exponent = exponent;
        return result;
    }
}

void Flow::NArrayCore::BackwardPow()
{
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] * Exponent * pow( Operands[0]->Data[i], Exponent - 1 );
        Operands[0]->Gradient->Data[i] += grad;
    }
}