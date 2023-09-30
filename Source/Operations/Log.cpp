// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Log( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
        {
            if ( value <= 0 ) return nullptr;
            else value = log(value);
        }
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::LOG );
    }
}

void Flow::NArrayCore::BackwardLog()
{
    NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        if ( operand->Data[i] == 0 ) return;
        float grad = Gradient->Data[i] / operand->Data[i];
        operand->Gradient->Data[i] += grad;
    }
}