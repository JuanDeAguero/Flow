// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index )
    {
        vector<float> resultData;
        vector<int> resultShape = index->GetShape();
        for ( int i = 0; i < index->Get().size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, resultShape );
            multiIndex[dim] = static_cast<int>(index->Get()[i]);
            int flatIndex = MultiToFlatIndex( multiIndex, arr->GetShape() );
            resultData.push_back(arr->Get()[flatIndex]);
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::GATHER );
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
    }
}

void Flow::NArrayCore::BackwardGather()
{
    for ( int i = 0; i < GatherIndex->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, GatherIndex->Shape );
        int indexElement = static_cast<int>(GatherIndex->Data[i]);
        multiIndex[GatherDim] = indexElement;
        int flatIndex = MultiToFlatIndex( multiIndex, Operands[0]->Shape );
        Operands[0]->Gradient->Data[flatIndex] += Gradient->Data[i];
    }
}