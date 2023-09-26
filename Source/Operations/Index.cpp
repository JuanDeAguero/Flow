// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Index( NArrayCore* arr, int dim, NArrayCore* index )
    {
        vector<int> arrShape = arr->GetShape();
        vector<float> indexData = index->Get();
        vector<int> resultShape = arrShape;
        resultShape[dim] = indexData.size();
        vector<float> resultData(resultShape[dim] * SizeFromShape(resultShape));
        int stride = arr->GetStride()[dim];
        for(int i = 0; i < indexData.size(); i++)
        {
            int idx = static_cast<int>(indexData[i]);
            for(int j = 0; j < stride; j++)
                resultData[i * stride + j] = arr->Get()[idx * stride + j];
        }

        NArrayCore* result = new NArrayCore(resultShape, resultData, { arr, index }, NArrayCore::Operation::INDEX);
        result->IndexDim = dim;
        result->Index = index;
        return result;
    }
}

void Flow::NArrayCore::BackwardIndex()
{
    NArrayCore* operand = Operands[0];
    vector<float> indexData = Index->Get();
    int stride = GetStride()[IndexDim];
    for(int i = 0; i < indexData.size(); i++)
    {
        int idx = static_cast<int>(indexData[i]);
        for(int j = 0; j < stride; j++)
            operand->Gradient->Data[idx * stride + j] += Gradient->Data[i * stride + j];
    }
}