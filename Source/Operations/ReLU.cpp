// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* ReLU( NArrayCore* arr )
    {
        throw runtime_error("Not implemented.");
        /*vector<float> data = arr->Get();
        for ( float& value : data )
            value = max( 0.0f, value );
        return new NArrayCore( arr->GetShape(), data, { arr }, NArrayCore::Operation::RELU );*/
    }
}

void Flow::NArrayCore::BackwardReLU()
{
    throw runtime_error("Not implemented.");
    /*NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = ( operand->Data[i] > 0.0f ) ? Gradient->Data[i] : 0.0f;
        operand->Gradient->Data[i] += grad;
    }*/
}