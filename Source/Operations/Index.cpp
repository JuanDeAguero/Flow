// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index )
    {
        vector<int> indices(index->Get().size());
        for ( int i = 0; i < index->Get().size(); i++ )
            indices[i] = static_cast<int>(index->Get()[i]);
        vector<int> resultShape = arr->GetShape();
        resultShape[dim] = indices.size();
        int resultSize = 1;
        for ( int size : resultShape ) resultSize *= size;
        vector<float> resultData(resultSize);
        for ( int i = 0; i < resultSize; i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, resultShape );
            multiIndex[dim] = indices[multiIndex[dim]];
            int flatIndex = MultiToFlatIndex( multiIndex, arr->GetShape() );
            resultData[i] = arr->Get()[flatIndex];
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr, index }, NArrayCore::Operation::INDEX );
        result->IndexDim = dim;
        result->Index = index;
        return result;
    }
}

void Flow::NArrayCore::BackwardIndex()
{
    vector<int> indices(Index->Get().size());
    for ( int i = 0; i < Index->Get().size(); i++ )
        indices[i] = static_cast<int>(Index->Get()[i]);
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, GetShape() );
        multiIndex[IndexDim] = indices[multiIndex[IndexDim]];
        int flatIndex = MultiToFlatIndex( multiIndex, Operands[0]->GetShape() );
        Operands[0]->Gradient->Data[flatIndex] += Gradient->Data[i];
    }
}