// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* Unsqueeze(NArrayCore* arr, int dim)
    {
        vector<int> shape = arr->GetShape();
        if (dim < 0)
            dim = static_cast<int>(shape.size());
        if (dim < 0 || dim > static_cast<int>(shape.size()))
            return nullptr;
        shape.insert(shape.begin() + dim, 1);
        NArrayCore* result = new NArrayCore(shape, arr->Get(), { arr }, NArrayCore::Operation::UNSQUEEZE);
        result->UnsqueezeDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardUnsqueeze()
{
    NArrayCore* operand = Operands[0];
    for (int i = 0; i < Data.size(); i++)
        operand->Gradient->Data[i] += Gradient->Data[i];
}