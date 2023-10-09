// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "ElementWise.hpp"
#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 )
    {
        auto shape = GetShapeForBroadcast( arr1, arr2 );
        Flow::NArrayCore* arr1B = Flow::Broadcast( arr1, shape );
        Flow::NArrayCore* arr2B = Flow::Broadcast( arr2, shape );
        auto op = NArrayCore::Operation::ADD;
        NArrayCore* result = new NArrayCore( arr1B->GetShape(), arr1B->Get(), { arr1B, arr2B }, op );
        vector<int> index = {};
        //ElementWise( index, arr1B, arr2B, result, op );
        ElementWise_CUDA( arr1B, arr2B, result, op );
        return result;
    }
}

void Flow::NArrayCore::BackwardAdd()
{
    if ( Operands.size() != 2 )
        throw runtime_error("[BackwardAdd] Invalid number of operands.");
    NArrayCore* operand1 = Operands[0];
    NArrayCore* operand2 = Operands[1];
    if ( Gradient->Data.size() != operand1->Gradient->Data.size() ||
        Gradient->Data.size() != operand2->Gradient->Data.size() )
        throw runtime_error("[BackwardAdd] Invalid operand gradient.");
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        operand1->Gradient->Data[i] += Gradient->Data[i];
        operand2->Gradient->Data[i] += Gradient->Data[i];
    }
}