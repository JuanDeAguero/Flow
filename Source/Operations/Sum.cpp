// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Sum( NArrayCore* arr, int dim )
    {
        vector<int> arrShape = arr->GetShape();
        if ( dim < 0 || dim >= arrShape.size() )
            throw out_of_range("Invalid dimension for Sum operation.");

        vector<int> resultShape = arrShape;  // Start with original shape
        resultShape[dim] = 1;                // Change the dimension being summed to have size 1

        vector<float> resultData( SizeFromShape(resultShape), 0.0f );
        vector<float> arrData = arr->Get();

        for ( int i = 0; i < arrData.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arrShape );
            vector<int> resultMultiIndex = multiIndex;
            resultMultiIndex[dim] = 0;       // Set the dim to 0 for the result

            int outIndex = MultiToFlatIndex( resultMultiIndex, resultShape );
            resultData[outIndex] += arrData[i];
        }

        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::SUM );
        result->SumDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardSum()
{
    NArrayCore* operand = Operands[0];
    vector<int> operandShape = operand->GetShape();

    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, GetShape() );

        for ( int j = 0; j < operandShape[SumDim]; j++ )
        {
            multiIndex[SumDim] = j;
            int flatIndex = MultiToFlatIndex( multiIndex, operandShape );
            operand->Gradient->Data[flatIndex] += Gradient->Data[i];
        }
    }
}