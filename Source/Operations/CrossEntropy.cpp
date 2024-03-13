// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::CrossEntropy( NARRAY arr1, NARRAY arr2 )
{
    NARRAY small = Create( { 1 }, { 1e-10f } );
    NARRAY arrUnsqueezed2 = Unsqueeze( arr2, 1 );
    return Mean( Neg( Log( Add( Gather( Softmax( arr1, 1 ), 1, arrUnsqueezed2 ), small ) ) ), 0 );
}