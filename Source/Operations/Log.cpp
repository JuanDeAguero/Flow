// Copyright (c) 2023 Juan M. G. de Agüero

#include <cmath>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Log( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = log(value);
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::LOG );
    }
}

void Flow::NArrayCore::BackwardLog()
{
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = Gradient->Data[i] / Operands[0]->Data[i];
        Operands[0]->Gradient->Data[i] += grad;
    }
}