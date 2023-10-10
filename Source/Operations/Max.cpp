// Copyright (c) 2023 Juan M. G. de Agüero

#include <limits>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Max( NArrayCore* arr, int dim )
    {
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = 1;
        vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
        for ( int i = 0; i < arr->Get().size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arr->GetShape() );
            multiIndex[dim] = 0;
            int flatIndex = MultiToFlatIndex( multiIndex, resultShape );
            resultData[flatIndex] = max( resultData[flatIndex], arr->Get()[i] );
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::MAX );
        result->MaxDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardMax()
{
    for ( int i = 0; i < Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, Shape );
        for ( int j = 0; j < Operands[0]->GetShape()[MaxDim]; j++ )
        {
            multiIndex[MaxDim] = j;
            int flatIndex = MultiToFlatIndex( multiIndex, Operands[0]->GetShape() );
            if ( Operands[0]->Data[flatIndex] == Data[i] )
            {
                Operands[0]->Gradient->Data[flatIndex] += Gradient->Data[i];
                break;
            }
        }
    }
}