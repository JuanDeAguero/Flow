// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Unsqueeze( NARRAY arr, int dim )
{
    vector<int> resultShape = arr->Shape;
    resultShape.insert( resultShape.begin() + dim, 1 );
    NARRAY result = make_shared<NArray>( resultShape, Flow::StrideFromShape(resultShape),
        arr->GetOffset(), FindMetaParent(arr), vector<NARRAY>{arr}, NArray::Operation::UNSQUEEZE );
    result->UnsqueezeDim = dim;
    return result;
}

void Flow::NArray::BackwardUnsqueeze()
{
    Operands[0]->Gradient = Flow::Squeeze( Gradient->Copy(), UnsqueezeDim );
}