// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Sum( NArrayCore* arr, int dim )
    {
        vector<int> arrShape = arr->GetShape();
        if ( dim < 0 || dim >= arrShape.size() )
            throw runtime_error("[Sum] Invalid dimension.");
        vector<int> resultShape = arrShape;
        resultShape[dim] = 1;
        vector<float> arrData = arr->Get();
        vector<float> resultData( SizeFromShape(resultShape), 0.0f );
        for ( int i = 0; i < arrData.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arrShape );
            multiIndex[dim] = 0;
            int flatIndex = MultiToFlatIndex( multiIndex, resultShape );
            resultData[flatIndex] += arrData[i];
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
        vector<int> multiIndex = FlatToMultiIndex( i, Shape );
        for ( int j = 0; j < operandShape[SumDim]; j++ )
        {
            multiIndex[SumDim] = j;
            int flatIndex = MultiToFlatIndex( multiIndex, operandShape );
            operand->Gradient->Data[flatIndex] += Gradient->Data[i];
        }
    }
}