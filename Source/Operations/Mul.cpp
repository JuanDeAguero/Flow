// Copyright (c) 2023 Juan M. G. de Agüero

#include "ElementWise.hpp"
#include "Flow/NArrayCore.h"

namespace Flow
{   
    NArrayCore* Mul( NArrayCore* arr1, NArrayCore* arr2 )
    {
        auto shape = GetShapeForBroadcast( arr1, arr2 );
        Flow::NArrayCore* arr1B = Flow::Broadcast( arr1, shape );
        Flow::NArrayCore* arr2B = Flow::Broadcast( arr2, shape );
        auto op = NArrayCore::Operation::MUL;
        NArrayCore* result = new NArrayCore( arr1B->GetShape(), arr1B->Get(), { arr1B, arr2B }, op );
        ElementWise( arr1B, arr2B, result, op );
        return result;
    }

    NArrayCore* Mul( NArrayCore* arr, float literal )
    {
        NArrayCore* arrLiteral = new NArrayCore( { 1 }, { literal } );
        return Mul( arr, arrLiteral );
    }
}

void Flow::NArrayCore::BackwardMul()
{
    if (UseCUDA)
    {
        BackwardMul_CUDA();
        return;
    }

    for ( int i = 0; i < Data.size(); i++ )
    {
        Operands[0]->Gradient->Data[i] += Operands[1]->Data[i] * Gradient->Data[i];
        Operands[1]->Gradient->Data[i] += Operands[0]->Data[i] * Gradient->Data[i];
    }
}