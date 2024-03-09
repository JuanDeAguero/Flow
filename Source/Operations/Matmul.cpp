// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Matmul( NARRAY arr1, NARRAY arr2 )
{
    int dim1 = arr1->GetShape().size();
    int dim2 = arr2->GetShape().size();
    if ( dim1 == 1 && dim2 == 1 ) return Sum( Mul( arr1, arr2 ), 0 );
    else if ( dim1 == 2 && dim2 == 1 )
    {
        NARRAY reshapedArr2 = Reshape( arr2, { arr1->GetShape()[0], 1 } );
        return Sum( Broadcast( Mul( arr1, reshapedArr2 ), arr1->GetShape()), 1 );
    }
    else if ( dim1 == 1 && dim2 == 2 ) return Squeeze( MM( Unsqueeze( arr1, 0 ), arr2 ), 0 );
    else if ( dim1 == 2 && dim2 == 2 ) return MM( arr1, arr2 );
    else if ( dim1 >= 3 && ( dim2 == 1 || dim2 == 2 ) )
    {

    }
    else if ( ( dim2 >= 1 && dim2 >= 1 ) && ( dim2 >= 3 || dim2 >= 3 ) )
    {
        
    }
    return Create( { 1 }, { 0.0f } );
}