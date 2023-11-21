// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {
        vector<int> shape = arr->GetShape();
        swap( shape[firstDim], shape[secondDim] );
        return new NArrayCore( shape, arr->Get(), { arr }, NArrayCore::Operation::RESHAPE );
    }
}

void Flow::NArrayCore::BackwardTranspose()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
}