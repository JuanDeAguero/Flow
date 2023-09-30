// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Unsqueeze( NArrayCore* arr, int dim )
    {
        vector<int> arrShape = arr->GetShape();
        if ( dim < 0 || dim > arrShape.size() )
            throw out_of_range("Invalid dimension for Unsqueeze operation.");
        arrShape.insert( arrShape.begin() + dim, 1 );
        vector<float> arrData = arr->Get();
        NArrayCore* result = new NArrayCore( arrShape, arrData, { arr }, NArrayCore::Operation::UNSQUEEZE );
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