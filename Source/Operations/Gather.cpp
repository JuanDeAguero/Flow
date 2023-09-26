// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    vector<int> FlatToMultiIndex(int flatIndex, const vector<int>& shape)
    {
        vector<int> multiIndex(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            multiIndex[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
        return multiIndex;
    }

    int MultiToFlatIndex(const vector<int>& multiIndex, const vector<int>& shape)
    {
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            flatIndex += multiIndex[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

    NArrayCore* Gather( NArrayCore* arr, int dim, NArrayCore* index )
    {
        if (arr->GetShape().size() != index->GetShape().size())
            return nullptr;
        if (dim < 0 || dim >= arr->GetShape().size())
            return nullptr;
        for (int d = 0; d < arr->GetShape().size(); d++)
        {
            if (d != dim && index->GetShape()[d] > arr->GetShape()[d])
                return nullptr;
        }
        vector<float> resultData;
        vector<int> shape = index->GetShape();
        const vector<float>& arrData = arr->Get();
        const vector<float>& indexData = index->Get();
        for (int flat_idx = 0; flat_idx < indexData.size(); ++flat_idx)
        {
            vector<int> multiIndex = FlatToMultiIndex(flat_idx, shape);
            int idx = static_cast<int>(indexData[flat_idx]);
            if (idx >= arr->GetShape()[dim])
                return nullptr;
            multiIndex[dim] = idx;
            int dataIdx = MultiToFlatIndex(multiIndex, arr->GetShape());
            resultData.push_back(arrData[dataIdx]);
        }
        NArrayCore* result = new NArrayCore(shape, resultData, { arr }, NArrayCore::Operation::GATHER);
        result->GatherDim = dim;
        result->GatherIndex = index;
        return result;
    }
}

void Flow::NArrayCore::BackwardGather()
{
    NArrayCore* operand = Operands[0];

    for (int flat_idx = 0; flat_idx < GatherIndex->Data.size(); ++flat_idx)
    {
        vector<int> multiIndex = FlatToMultiIndex(flat_idx, GatherIndex->Shape);
        int gather_idx = static_cast<int>(GatherIndex->Data[flat_idx]);
        multiIndex[GatherDim] = gather_idx;
        int dataIdx = MultiToFlatIndex(multiIndex, operand->Shape);
        operand->Gradient->Data[dataIdx] += Gradient->Data[flat_idx];
    }
}