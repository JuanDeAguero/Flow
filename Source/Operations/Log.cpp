// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Log( NArrayCore* arr )
    {
        vector<float> resultData = arr->Get();
        
        // Forward computation of the natural logarithm for each element in the array
        for ( float& value : resultData )
        {
            if (value <= 0) 
            {
                // Error handling - Log of non-positive values is undefined
                //throw std::runtime_error("Attempted to compute log of non-positive value.");
            }
            value = log(value);
        }

        // Store the result in a new NArrayCore object with the TANH operation (assuming there's a LOG operation enum value)
        NArrayCore* result = new NArrayCore( arr->GetShape(), resultData, { arr }, NArrayCore::Operation::LOG );
        return result;
    }
}

void Flow::NArrayCore::BackwardLog()
{
    NArrayCore* operand = Operands[0];

    // Compute the gradient for the logarithm function
    for ( int i = 0; i < Data.size(); i++ )
    {
        // The derivative of log(x) is 1/x
        if (operand->Data[i] == 0)
        {
            // Error handling - derivative at 0 is undefined
            //throw std::runtime_error("Attempted to compute derivative of log at value 0.");
        }
        float grad = Gradient->Data[i] / operand->Data[i];
        operand->Gradient->Data[i] += grad;
    }
}