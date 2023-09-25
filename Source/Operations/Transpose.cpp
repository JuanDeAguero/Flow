// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

using namespace std;

static vector<int> TransposeStride(const vector<int>& shape)
{
    vector<int> stride(shape.size());
    int s = 1;
    for(int i = shape.size() - 1; i >= 0; i--) {
        stride[i] = s;
        s *= shape[i];
    }
    return stride;
}

namespace Flow
{
    NArrayCore* Transpose( NArrayCore* arr, int firstDim, int secondDim )
    {
        vector<int> originalShape = arr->GetShape();
        vector<float> originalData = arr->Get();
        swap(originalShape[firstDim], originalShape[secondDim]);
        vector<int> transposedStride = TransposeStride(originalShape);
        vector<float> transposedData(originalData.size());
        int totalDims = originalShape.size();
        for (int i = 0; i < originalData.size(); i++)
        {
            vector<int> coords(totalDims, 0);
            int tempIdx = i;
            for (int d = 0; d < totalDims; d++)
            {
                coords[d] = tempIdx / arr->GetStride()[d];
                tempIdx %= arr->GetStride()[d];
            }
            swap(coords[firstDim], coords[secondDim]);
            int newIndex = 0;
            for (int d = 0; d < totalDims; d++)
            {
                newIndex += coords[d] * transposedStride[d];
            }
            transposedData[newIndex] = originalData[i];
        }
        NArrayCore* result = new NArrayCore(originalShape, transposedData, { arr }, NArrayCore::Operation::TRANSPOSE);
        result->TransposeFirstDim = firstDim;
        result->TransposeSecondDim = secondDim;
        return result;
    }
}

void Flow::NArrayCore::BackwardTranspose()
{
    NArrayCore* operand = Operands[0];
    int firstDim = TransposeFirstDim;
    int secondDim = TransposeSecondDim;
    int totalDims = Shape.size();
    vector<int> operandShape = operand->GetShape();
    vector<int> operandStride = TransposeStride(operandShape);
    for (int i = 0; i < Data.size(); i++)
    {
        vector<int> coords(totalDims, 0);
        int tempIdx = i;
        for (int d = 0; d < totalDims; d++)
        {
            coords[d] = tempIdx / Stride[d];
            tempIdx %= Stride[d];
        }
        swap(coords[firstDim], coords[secondDim]);
        int originalIndex = 0;
        for (int d = 0; d < totalDims; d++)
        {
            originalIndex += coords[d] * operandStride[d];
        }
        operand->Gradient->Data[originalIndex] += Gradient->Data[i];
    }
}