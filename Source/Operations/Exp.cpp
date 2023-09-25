// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Exp( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = exp(value);
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::EXP );
        return result;
    }
}

void Flow::NArrayCore::BackwardExp()
{
    NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] * exp(operand->Data[i]);
        operand->Gradient->Data[i] += grad;
    }
}