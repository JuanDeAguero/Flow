// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Conv2d( NARRAY arr, NARRAY weight )
{
    int batchSize = arr->GetShape()[0];
    int inSize = arr->GetShape()[2];
    int outChannels = weight->GetShape()[0];
    int inChannels = weight->GetShape()[1];
    int kernelSize = weight->GetShape()[2];
    int outSize = inSize - kernelSize + 1;
    NARRAY unfolded = Flow::Unfold2d( arr, { kernelSize, kernelSize } );
    NARRAY unfoldedTransposed = Transpose( unfolded, 1, 2 );
    auto weightShape = { outChannels, inChannels * kernelSize * kernelSize };
    NARRAY weightTransposed = Transpose( Reshape( weight, weightShape ), 0, 1 );
    NARRAY conv = Flow::Transpose( Matmul( unfoldedTransposed, weightTransposed ), 1, 2 );
    NARRAY folded = Flow::Fold2d( conv, { batchSize, outChannels, outSize, outSize }, { 1, 1 } );
    return folded;
}