// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>
#include <string>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Log( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
        {
            if ( value <= 0 )
                throw runtime_error( "[Log] Invalid value: " + to_string(value) );
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
        if ( operand->Data[i] == 0 )
            throw runtime_error( "[BackwardLog] Invalid zero operand." );
        float grad = Gradient->Data[i] / operand->Data[i];
        operand->Gradient->Data[i] += grad;
    }
}