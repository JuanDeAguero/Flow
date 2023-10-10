// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Unsqueeze( NArrayCore* arr, int dim )
    {
        vector<int> resultShape = arr->GetShape();
        resultShape.insert( resultShape.begin() + dim, 1 );
        NArrayCore* result = new NArrayCore( resultShape, arr->Get(), { arr }, NArrayCore::Operation::UNSQUEEZE );
        result->UnsqueezeDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardUnsqueeze()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
}