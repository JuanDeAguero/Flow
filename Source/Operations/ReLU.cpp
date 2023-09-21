// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* ReLU( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();

        // Forward computation of the ReLU for each element in the array
        for ( float& value : resultData )
            value = std::max(0.0f, value);

        // Store the result in a new NArrayCore object with the appropriate operation tag (assuming there's a RELU operation enum value)
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::RELU );
        return result;
    }
}

void Flow::NArrayCore::BackwardReLU()
{
    NArrayCore* operand = Operands[0];

    // Compute the gradient for the ReLU function
    for ( int i = 0; i < Data.size(); i++ )
    {
        // The derivative is 1 where the original input was positive, and 0 otherwise
        float grad = (operand->Data[i] > 0.0f) ? Gradient->Data[i] : 0.0f;
        operand->Gradient->Data[i] += grad;
    }
}