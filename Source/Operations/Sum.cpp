// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Sum( NArrayCore* arr, int dim, bool keepDim )
    {
        vector<int> inputShape = arr->GetShape();
        int numDims = inputShape.size();

        // If dim is -1, sum over all dimensions
        if(dim == -1)
        {
            float totalSum = 0.0f;
            for(float value : arr->Get())
                totalSum += value;
            return new NArrayCore({1}, {totalSum});  // Return a tensor with single value
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
        vector<float> outputData(outputSize, 0.0f);

        for(int i = 0; i < arr->Get().size(); i += stride)
        {
            float sum = 0.0f;
            for(int j = 0; j < dimSize; ++j)
                sum += arr->Get()[i + j];
            
            outputData[i / dimSize] = sum;
        }

        NArrayCore* result = new NArrayCore(outputShape, outputData, {arr}, NArrayCore::Operation::SUM);
        result->Dim = dim;
        result->KeepDim = keepDim;
        return result;
    }
}

void Flow::NArrayCore::BackwardSum()
{
    // Get the gradient from the output tensor (dL/dy, where y is the output)
    NArrayCore* outGrad = this->Gradient;

    // Create a gradient tensor for the input (dL/dx, where x is the input)
    vector<float> inGradData(this->Data.size(), 0.0f);
    NArrayCore* inGrad = new NArrayCore(this->Shape, inGradData, true);

    // If the sum was done over all dimensions:
    if (this->Dim == -1)
    {
        // Each element in the input tensor contributes equally to the sum,
        // so the gradient for each element is just the gradient of the output.
        float scalarGrad = outGrad->Get()[0];
        for (float& val : inGrad->Data)
        {
            val = scalarGrad;
        }
    }
    else
    {
        // If sum was done along a specific dimension:
        int stride = this->Stride[this->Dim];
        int dimSize = this->Shape[this->Dim];
        int indexMultiplier = this->KeepDim ? dimSize : 1;

        for (int i = 0; i < inGrad->Data.size(); i += stride)
        {
            // For each slice along the summed dimension, all elements contributed equally.
            // Therefore, each element in the slice has the same gradient.
            float sliceGrad = outGrad->Get()[i / indexMultiplier];
            for (int j = 0; j < dimSize; ++j)
            {
                inGrad->Data[i + j] = sliceGrad;
            }
        }
    }

    // Assign the computed gradient to the operands
    for (NArrayCore* operand : this->Operands)
    {
        for (int i = 0; i < operand->Data.size(); ++i)
        {
            operand->Gradient->Data[i] += inGrad->Data[i];
        }
    }

    // Clean up
    delete inGrad;
}