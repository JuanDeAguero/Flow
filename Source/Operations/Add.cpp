// Copyright (c) 2023 Juan M. G. de Agüero

#include "ElementWise.hpp"
#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 )
    {
        auto shape = GetShapeForBroadcast( arr1, arr2 );
        Flow::NArrayCore* arr1B = Flow::Broadcast( arr1, shape );
        Flow::NArrayCore* arr2B = Flow::Broadcast( arr2, shape );
        auto op = NArrayCore::Operation::ADD;
        NArrayCore* result = new NArrayCore( arr1B->GetShape(), arr1B->Get(), { arr1B, arr2B }, op );
        ElementWise( arr1B, arr2B, result, op );
        return result;
    }
}

void Flow::NArrayCore::BackwardAdd()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
        Operands[1]->Gradient->Data[i] += Gradient->Data[i];
    }
}