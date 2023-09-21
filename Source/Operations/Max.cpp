// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Max( NArrayCore* arr, int dim, bool keepDim )
    {
        vector<int> inputShape = arr->GetShape();
        int numDims = inputShape.size();

        // If dim is -1, find the maximum over all dimensions
        if(dim == -1)
        {
            float globalMax = std::numeric_limits<float>::lowest();
            for(float value : arr->Get())
                globalMax = std::max(globalMax, value);
            return new NArrayCore({1}, {globalMax});  // Return a tensor with single value
        }

        // Ensure the dimension is valid
        if(dim < 0 || dim >= numDims)
            return nullptr;  // Invalid dimension

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
        result->Dim = dim;
        result->KeepDim = keepDim;
        return result;
    }
}

void Flow::NArrayCore::BackwardMax()
{
    // Check if the operand is empty or the gradient hasn't been set. If yes, just return.
    if (Operands.size() == 0 || !Gradient)
        return;

    // Get the gradient data from the result of the Max operation.
    vector<float> outputGradient = Gradient->Get();
    
    // Get the data and shape from the input tensor.
    vector<float> inputData = Operands[0]->Get();
    vector<int> inputShape = Operands[0]->GetShape();
    int stride = Operands[0]->GetStride()[Dim];
    int dimSize = inputShape[Dim];

    // Create a gradient tensor for the input with the same shape as the input tensor and initialize with zeros.
    vector<float> inputGradient(inputData.size(), 0.0f);

    for (int i = 0; i < inputData.size(); i += stride)
    {
        // Get the maximum value and its index along the specified dimension.
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

        // The gradient of the max operation is 1 for the max value and 0 for all others.
        // Thus, propagate the gradient from the output of the Max operation to the maximum value in the input tensor.
        inputGradient[i + maxIndex] = outputGradient[i / dimSize];
    }

    // Update the gradient of the input tensor.
    Operands[0]->Gradient->Reset(0.0f);
    for (int i = 0; i < inputData.size(); ++i)
    {
        Operands[0]->Gradient->Set({i}, inputGradient[i]);
    }
}