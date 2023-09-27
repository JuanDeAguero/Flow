// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    static vector<int> FlatToMultiIndex( int index, vector<int> shape )
    {
        vector<int> multiIndex(shape.size());
        for ( int i = shape.size() - 1; i >= 0; i-- )
        {
            multiIndex[i] = index % shape[i];
            index /= shape[i];
        }
        return multiIndex;
    }

    static int MultiToFlatIndex( vector<int> index, vector<int> shape )
    {
        int flatIndex = 0;
        int stride = 1;
        for ( int i = shape.size() - 1; i >= 0; i-- )
        {
            flatIndex += index[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

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
        vector<float> data;
        vector<int> shape = index->GetShape();
        vector<float> arrData = arr->Get();
        vector<float> indexData = index->Get();
        for ( int i = 0; i < indexData.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, shape );
            int indexElement = static_cast<int>(indexData[i]);
            if ( indexElement >= arr->GetShape()[dim] )
                return nullptr;
            multiIndex[dim] = indexElement;
            int flatIndex = MultiToFlatIndex( multiIndex, arr->GetShape() );
            data.push_back(arrData[flatIndex]);
        }
        NArrayCore* result = new NArrayCore( shape, data, { arr }, NArrayCore::Operation::GATHER );
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