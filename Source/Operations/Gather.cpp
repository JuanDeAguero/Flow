// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index )
    {
        if (UseCUDA)
            return Gather_CUDA( arr, dim, index );

        vector<float> resultData;
        for ( int i = 0; i < index->Get().size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, index->GetShape() );
            multiIndex[dim] = (int)(index->Get()[i]);
            int flatIndex = MultiToFlatIndex( multiIndex, arr->GetShape() );
            resultData.push_back(arr->Get()[flatIndex]);
        }
        NArrayCore* result = new NArrayCore( index->GetShape(), resultData, { arr }, NArrayCore::Operation::GATHER );
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
    }

    void NArrayCore::BackwardGather()
    {
        if (UseCUDA)
        {
            BackwardGather_CUDA();
            return;
        }

        for ( int i = 0; i < GatherIndex->Data.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, GatherIndex->Shape );
            int indexElement = (int)(GatherIndex->Data[i]);
            multiIndex[GatherDim] = indexElement;
            int flatIndex = MultiToFlatIndex( multiIndex, Operands[0]->Shape );
            Operands[0]->Gradient->Data[flatIndex] += Gradient->Data[i];
        }
    }
}