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

}