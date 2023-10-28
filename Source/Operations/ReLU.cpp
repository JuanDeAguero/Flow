// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* ReLU( NArrayCore* arr )
    {
        if (UseCUDA)
            return ReLU_CUDA(arr);

        vector<float> resultData = arr->Get();
        for ( float& value : resultData )
            value = max( 0.0f, value );
        return new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::RELU );
    }
}

void Flow::NArrayCore::BackwardReLU()
{
    if (UseCUDA)
    {
        BackwardReLU_CUDA();
        return;
    }

    for ( int i = 0; i < Data.size(); i++ )
    {
        float grad = ( Operands[0]->Data[i] > 0.0f ) ? Gradient->Data[i] : 0.0f;
        Operands[0]->Gradient->Data[i] += grad;
    }
}