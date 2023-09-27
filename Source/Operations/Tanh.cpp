// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Tanh( NArrayCore* arr )
    {
        throw runtime_error("Not implemented.");
        /*vector<float> data = arr->Get();
        for ( float& value : data )
            value = tanh(value);
        return new NArrayCore( arr->GetShape(), data, { arr }, NArrayCore::Operation::TANH );*/
    }
}

void Flow::NArrayCore::BackwardTanh()
{
    throw runtime_error("Not implemented.");
    /*NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float value = tanh(operand->Data[i]);
        float grad = Gradient->Data[i] * ( 1 - value * value );
        operand->Gradient->Data[i] += grad;
    }*/
}