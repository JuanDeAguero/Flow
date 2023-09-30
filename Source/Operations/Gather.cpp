// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index )
    {
        if ( arr->GetShape().size() != index->GetShape().size() )
            return nullptr;
        if ( dim < 0 || dim >= arr->GetShape().size() )
            return nullptr;
        for ( int i = 0; i < arr->GetShape().size(); i++ )
        {
            if ( i != dim && index->GetShape()[i] > arr->GetShape()[i] )
                return nullptr;
        }
        vector<float> resultData;
        vector<int> resultShape = index->GetShape();
        vector<float> arrData = arr->Get();
        vector<float> indexData = index->Get();
        for ( int i = 0; i < indexData.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, resultShape );
            int indexElement = static_cast<int>(indexData[i]);
            if ( indexElement >= arr->GetShape()[dim] )
                return nullptr;
            multiIndex[dim] = indexElement;
            int flatIndex = MultiToFlatIndex( multiIndex, arr->GetShape() );
            resultData.push_back(arrData[flatIndex]);
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::GATHER );
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
    }
}

void Flow::NArrayCore::BackwardGather()
{
    NArrayCore* operand = Operands[0];
    for ( int i = 0; i < GatherIndex->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, GatherIndex->Shape );
        int indexElement = static_cast<int>(GatherIndex->Data[i]);
        multiIndex[GatherDim] = indexElement;
        int flatIndex = MultiToFlatIndex( multiIndex, operand->Shape );
        operand->Gradient->Data[flatIndex] += Gradient->Data[i];
    }
}