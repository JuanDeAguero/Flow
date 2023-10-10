// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Tanh( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = tanh(value);
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::TANH );
    }
}

void Flow::NArrayCore::BackwardTanh()
{
    for ( int i = 0; i < Data.size(); i++ )
    {
        float value = tanh(Operands[0]->Data[i]);
        float grad = Gradient->Data[i] * ( 1 - value * value );
        Operands[0]->Gradient->Data[i] += grad;
    }
}