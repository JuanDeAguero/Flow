// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Sum( NArrayCore* arr, int dim, bool keepDim )
    {
        vector<int> inputShape = arr->GetShape();
        int numDims = inputShape.size();
        if(dim == -1)
        {
            float totalSum = 0.0f;
            for(float value : arr->Get())
                totalSum += value;
            return new NArrayCore({1}, {totalSum}, {arr}, NArrayCore::Operation::SUM);
        }
        if(dim < 0 || dim >= numDims)
            return nullptr;
        int stride = arr->GetStride()[dim];
        int dimSize = inputShape[dim];
        vector<int> outputShape;
        for(int i = 0; i < numDims; ++i)
        {
            if(i != dim || keepDim)
                outputShape.push_back(inputShape[i]);
        }
        int outputSize = keepDim ? stride : stride * dimSize;
        vector<float> outputData(outputSize, 0.0f);
        for(int i = 0; i < arr->Get().size(); i += stride)
        {
            float sum = 0.0f;
            for(int j = 0; j < dimSize; ++j)
                sum += arr->Get()[i + j];
            
            outputData[i / dimSize] = sum;
        }
        NArrayCore* result = new NArrayCore(outputShape, outputData, {arr}, NArrayCore::Operation::SUM);
        result->SumDim = dim;
        result->SumKeepDim = keepDim;
        return result;
    }
}

void Flow::NArrayCore::BackwardSum()
{
    NArrayCore* outGrad = this->Gradient;
    vector<float> inGradData(this->Data.size(), 0.0f);
    NArrayCore* inGrad = new NArrayCore(this->Shape, inGradData, true);
    if (this->SumDim == -1)
    {
        float scalarGrad = outGrad->Get()[0];
        for (float& val : inGrad->Data)
            val = scalarGrad;
    }
    else
    {
        int stride = this->Stride[this->SumDim];
        int dimSize = this->Shape[this->SumDim];
        int indexMultiplier = this->SumKeepDim ? dimSize : 1;
        for (int i = 0; i < inGrad->Data.size(); i += stride)
        {
            float sliceGrad = outGrad->Get()[i / indexMultiplier];
            for (int j = 0; j < dimSize; ++j)
                inGrad->Data[i + j] = sliceGrad;
        }
    }
    for (NArrayCore* operand : this->Operands)
    {
        for (int i = 0; i < operand->Data.size(); ++i)
            operand->Gradient->Data[i] += inGrad->Data[i];
    }
    delete inGrad;
}