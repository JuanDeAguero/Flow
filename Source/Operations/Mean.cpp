// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Mean( NARRAY arr, int dim )
{
    NARRAY sum = Sum( arr, dim );
    NARRAY n = Create( { 1 }, { (float)arr->GetShape()[dim] } );
    return Div( sum, n );
}