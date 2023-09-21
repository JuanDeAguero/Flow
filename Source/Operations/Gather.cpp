// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Gather( NArrayCore* arr1, NArrayCore* arr2 )
    {
        // Check dimensions and ensure compatibility
        // Assuming arr1 is the tensor from which values are picked and arr2 contains the indices

        vector<int> inputShape = arr1->GetShape();
        vector<int> indexShape = arr2->GetShape();
        if (inputShape.size() < 1 || indexShape.size() < 1)
            return nullptr; // Invalid tensors
        
        vector<float> inputData = arr1->Get();
        vector<float> indexData = arr2->Get(); // Assuming indices are also stored as floats (not ideal, but for demonstration purposes)

        vector<float> outputData(indexShape[0]); // Assuming 1D indexing for simplicity

        for (int i = 0; i < indexShape[0]; i++) {
            int index = static_cast<int>(indexData[i]);
            if (index < 0 || index >= inputShape[0])
                return nullptr; // Invalid index
            outputData[i] = inputData[index];
        }

        return new NArrayCore(indexShape, outputData, {arr1, arr2}, NArrayCore::Operation::GATHER);
    }
}

void Flow::NArrayCore::BackwardGather()
{
    NArrayCore* inputTensor = Operands[0];
    NArrayCore* indexTensor = Operands[1];

    vector<float> inputGradient(inputTensor->Data.size(), 0.0f);
    vector<float> indexData = indexTensor->Get();

    for (int i = 0; i < Data.size(); i++) {
        int index = static_cast<int>(indexData[i]);
        inputGradient[index] += Gradient->Data[i]; // Add gradient to the appropriate location
    }

    for (int i = 0; i < inputGradient.size(); i++) {
        inputTensor->Gradient->Data[i] += inputGradient[i];
    }

    // Note: Gradient with respect to indices (arr2) is not well-defined for gather operation
    // Hence, we don't update arr2's gradient
}