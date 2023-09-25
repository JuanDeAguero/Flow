// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Max( NArrayCore* arr, int dim, bool keepDim )
    {
        vector<int> inputShape = arr->GetShape();
        int numDims = inputShape.size();
        if(dim == -1)
        {
            float globalMax = std::numeric_limits<float>::lowest();
            for(float value : arr->Get())
                globalMax = std::max(globalMax, value);
            return new NArrayCore({1}, {globalMax});
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
        vector<float> outputData(outputSize, std::numeric_limits<float>::lowest());
        for(int i = 0; i < arr->Get().size(); i += stride)
        {
            float maxVal = std::numeric_limits<float>::lowest();
            for(int j = 0; j < dimSize; ++j)
                maxVal = std::max(maxVal, arr->Get()[i + j]);
            
            outputData[i / dimSize] = maxVal;
        }
        NArrayCore* result = new NArrayCore(outputShape, outputData, {arr}, NArrayCore::Operation::MAX);
        result->MaxDim = dim;
        result->MaxKeepDim = keepDim;
        return result;
    }
}

void Flow::NArrayCore::BackwardMax()
{
    if (Operands.size() == 0 || !Gradient)
        return;
    vector<float> outputGradient = Gradient->Get();
    vector<float> inputData = Operands[0]->Get();
    vector<int> inputShape = Operands[0]->GetShape();
    int stride = Operands[0]->GetStride()[MaxDim];
    int dimSize = inputShape[MaxDim];
    vector<float> inputGradient(inputData.size(), 0.0f);
    for (int i = 0; i < inputData.size(); i += stride)
    {
        float maxVal = std::numeric_limits<float>::lowest();
        int maxIndex = -1;
        for (int j = 0; j < dimSize; ++j)
        {
            if (inputData[i + j] > maxVal)
            {
                maxVal = inputData[i + j];
                maxIndex = j;
            }
        }
        inputGradient[i + maxIndex] = outputGradient[i / dimSize];
    }
    Operands[0]->Gradient->Reset(0.0f);
    for (int i = 0; i < inputData.size(); ++i)
        Operands[0]->Gradient->Set({i}, inputGradient[i]);
}