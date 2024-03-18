// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::MaxPool2d( NARRAY arr, vector<int> kernel )
{
    auto arrShape = arr->GetShape();
    int kernelHeight = kernel[0];
    int kernelWidth = kernel[1];
    int outHeight = ( ( arrShape[2] - kernelHeight ) / kernelHeight ) + 1;
    int outWidth = ( ( arrShape[3] - kernelWidth ) / kernelWidth ) + 1;
    NARRAY unfolded = Unfold2d( arr, { kernelHeight, kernelWidth }, { kernelHeight, kernelWidth } );
    NARRAY unfoldedReshaped = Reshape( unfolded,
        { arrShape[0], arrShape[1], kernelHeight * kernelWidth, outHeight * outWidth } );
    NARRAY pooled = Max( unfoldedReshaped, 2 );
    return Reshape( pooled, { arrShape[0], arrShape[1], outHeight, outWidth } );
}