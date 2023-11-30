// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {
        vector<int> resultShape = arr->GetShape();
        swap( resultShape[firstDim], resultShape[secondDim] );
        return new NArrayCore( resultShape, arr->Get(), { arr }, NArrayCore::Operation::RESHAPE );
    }
}

void Flow::NArrayCore::BackwardTranspose()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
}