// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Exp( NArrayCore* arr )
    {
        throw runtime_error("Not implemented.");
        /*vector<float> data = arr->Get();
        for ( float& value : data )
            value = exp(value);
        return new NArrayCore( arr->GetShape(), data, { arr }, NArrayCore::Operation::EXP );*/
    }
}

void Flow::NArrayCore::BackwardExp()
{
    throw runtime_error("Not implemented.");
    /*NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] * exp( operand->Data[i] );
        operand->Gradient->Data[i] += grad;
    }*/
}