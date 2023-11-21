// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Reshape( NArrayCore* arr, vector<int> shape )
    {
        return new NArrayCore( shape, arr->Get(), { arr }, NArrayCore::Operation::RESHAPE );
    }
}

void Flow::NArrayCore::BackwardReshape()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
}