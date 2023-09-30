// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Max( NArrayCore* arr, int dim )
    {
        vector<int> inputShape = arr->GetShape();
        if (dim < 0 || dim >= inputShape.size())
            throw out_of_range("Invalid dimension for Max operation.");

        // Only change the size of the dimension being reduced to 1 instead of removing it.
        vector<int> resultShape = inputShape;
        resultShape[dim] = 1;
        
        vector<float> arrData = arr->Get();
        vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
        
        for (int i = 0; i < arrData.size(); i++)
        {
            vector<int> multiIndex = FlatToMultiIndex(i, inputShape);
            vector<int> resultMultiIndex = multiIndex;
            resultMultiIndex[dim] = 0; // Only change to set the dim to 0.
            
            int resultIndex = MultiToFlatIndex(resultMultiIndex, resultShape);
            resultData[resultIndex] = max(resultData[resultIndex], arrData[i]);
        }
        
        NArrayCore* result = new NArrayCore(resultShape, resultData, { arr }, NArrayCore::Operation::MAX);
        result->MaxDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardMax()
{
    NArrayCore* operand = Operands[0];
    vector<int> inputShape = operand->GetShape();
    for (int i = 0; i < Data.size(); i++)
    {
        vector<int> resultMultiIndex = FlatToMultiIndex(i, GetShape());
        vector<int> checkMultiIndex = resultMultiIndex;
        for (int j = 0; j < inputShape[MaxDim]; j++)
        {
            checkMultiIndex[MaxDim] = j;
            int inputIndex = MultiToFlatIndex(checkMultiIndex, inputShape);
            if (operand->Data[inputIndex] == Data[i])
            {
                operand->Gradient->Data[inputIndex] += Gradient->Data[i];
                break;
            }
        }
    }
}