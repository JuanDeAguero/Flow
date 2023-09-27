// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Pow( NArrayCore* arr, float exponent )
    {
        throw runtime_error("Not implemented.");
        /*vector<float> data = arr->Get();
        for ( float& value : data )
            value = pow( value, exponent );
        NArrayCore* result = new NArrayCore( arr->GetShape(), data, { arr }, NArrayCore::Operation::POW );
        result->Exponent = exponent;
        return result;*/
    }
}

void Flow::NArrayCore::BackwardPow()
{
    throw runtime_error("Not implemented.");
    /*NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] * Exponent * pow( operand->Data[i], Exponent - 1 );
        operand->Gradient->Data[i] += grad;
    }*/
}