// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* ReLU( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = max( 0.0f, value );
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::RELU );
        return result;
    }
}

void Flow::NArrayCore::BackwardReLU()
{
    NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = (operand->Data[i] > 0.0f) ? Gradient->Data[i] : 0.0f;
        operand->Gradient->Data[i] += grad;
    }
}