// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Sum( NArrayCore* arr, int dim )
    {
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = 1;
        vector<float> resultData( SizeFromShape(resultShape), 0.0f );
        for ( int i = 0; i < arr->Get().size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arr->GetShape() );
            multiIndex[dim] = 0;
            int flatIndex = MultiToFlatIndex( multiIndex, resultShape );
            resultData[flatIndex] += arr->Get()[i];
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::SUM );
        result->SumDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardSum()
{
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, Shape );
        for ( int j = 0; j < Operands[0]->GetShape()[SumDim]; j++ )
        {
            multiIndex[SumDim] = j;
            int flatIndex = MultiToFlatIndex( multiIndex, Operands[0]->GetShape() );
            Operands[0]->Gradient->Data[flatIndex] += Gradient->Data[i];
        }
    }
}