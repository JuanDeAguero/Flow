// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Squeeze( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->Shape;
    resultShape.erase( resultShape.begin() + dim );
    NARRAY result = make_shared<NArray>( resultShape, Flow::StrideFromShape(resultShape),
        arr->GetOffset(), FindMetaParent(arr), vector<NARRAY>{arr}, NArray::Operation::SQUEEZE );
    result->SqueezeShape = arr->Shape;
    return result;
}

void Flow::NArray::BackwardSqueeze()
{
    Operands[0]->Gradient = Reshape( Gradient->Copy(), SqueezeShape );
}