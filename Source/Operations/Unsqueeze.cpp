// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Unsqueeze( NArrayCore* arr, int dim )
    {
        if ( dim < 0 || dim > arr->GetShape().size() )
            throw runtime_error("[Unsqueeze] Invalid dimension.");
        vector<int> resultShape = arr->GetShape();
        resultShape.insert( resultShape.begin() + dim, 1 );
        NArrayCore* result = new NArrayCore( resultShape, arr->Get(), { arr }, NArrayCore::Operation::UNSQUEEZE );
        result->UnsqueezeDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardUnsqueeze()
{
    NArrayCore* operand = Operands[0];
    for ( int i = 0; i < Gradient->Data.size(); i++ )
        operand->Gradient->Data[i] += Gradient->Data[i];
}