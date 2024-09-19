// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Reshape( NARRAY arr, vector<int> shape )
{
    NARRAY result = make_shared<NArray>( shape, StrideFromShape(shape), arr->GetOffset(),
        FindMetaParent(arr), vector<NARRAY>{arr}, NArray::Operation::RESHAPE );
    result->ReshapeShape = arr->Shape;
    return result;
}

void Flow::NArray::BackwardReshape()
{
    Operands[0]->Gradient = Reshape( Gradient->Copy(), ReshapeShape );
}