// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {
        if (UseCUDA)
            return Transpose_CUDA( arr, firstDim, secondDim );

        vector<int> resultShape = arr->GetShape();
        int temp = resultShape[firstDim];
        resultShape[firstDim] = resultShape[secondDim];
        resultShape[secondDim] = temp;
        vector<float> resultData( arr->Get().size(), 0 );
        for ( int i = 0; i < arr->Get().size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arr->GetShape() );
            int temp = multiIndex[firstDim];
            multiIndex[firstDim] = multiIndex[secondDim];
            multiIndex[secondDim] = temp;
            int flatIndex = MultiToFlatIndex( multiIndex, resultShape );
            resultData[flatIndex] = arr->Get()[i];
        }
        return new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::TRANSPOSE );
    }
}

void Flow::NArrayCore::BackwardTranspose()
{

}