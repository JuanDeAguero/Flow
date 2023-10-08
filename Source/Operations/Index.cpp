// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index )
    {
        if ( index->GetShape().size() != 1 )
            throw runtime_error("[Index] The index must be 1D.");
        vector<float> indexData = index->Get();
        vector<int> indices(indexData.size());
        for ( int i = 0; i < indexData.size(); i++ )
        {
            if ( indexData[i] != static_cast<int>(indexData[i]) )
                throw runtime_error("[Index] All indices must be integers.");
            indices[i] = static_cast<int>(indexData[i]);
            if ( indices[i] < 0 || indices[i] >= arr->GetShape()[dim] )
                throw runtime_error("[Index] Index out of bounds for dimension specified.");
        }
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
    NArrayCore* operand = Operands[0];
    vector<float> indexData = Index->Get();
    vector<int> indices(indexData.size());
    for ( int i = 0; i < indexData.size(); i++ )
        indices[i] = static_cast<int>(indexData[i]);
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, GetShape() );
        multiIndex[IndexDim] = indices[multiIndex[IndexDim]];
        int flatIndex = MultiToFlatIndex( multiIndex, operand->GetShape() );
        operand->Gradient->Data[flatIndex] += Gradient->Data[i];
    }
}