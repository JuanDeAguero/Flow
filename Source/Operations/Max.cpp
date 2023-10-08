// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Max( NArrayCore* arr, int dim )
    {
        vector<int> arrShape = arr->GetShape();
        if ( dim < 0 || dim >= arrShape.size() )
            throw runtime_error("[Max] Invalid dimension.");
        vector<int> resultShape = arrShape;
        resultShape[dim] = 1;
        vector<float> arrData = arr->Get();
        vector<float> resultData( SizeFromShape(resultShape), numeric_limits<float>::min() );
        for ( int i = 0; i < arrData.size(); i++ )
        {
            vector<int> multiIndex = FlatToMultiIndex( i, arrShape );
            multiIndex[dim] = 0;
            int flatIndex = MultiToFlatIndex( multiIndex, resultShape );
            resultData[flatIndex] = max( resultData[flatIndex], arrData[i] );
        }
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr }, NArrayCore::Operation::MAX );
        result->MaxDim = dim;
        return result;
    }
}

void Flow::NArrayCore::BackwardMax()
{
    NArrayCore* operand = Operands[0];
    vector<int> operandShape = operand->GetShape();
    for ( int i = 0; i < Data.size(); i++ )
    {
        vector<int> multiIndex = FlatToMultiIndex( i, Shape );
        for ( int j = 0; j < operandShape[MaxDim]; j++ )
        {
            multiIndex[MaxDim] = j;
            int flatIndex = MultiToFlatIndex( multiIndex, operandShape );
            if ( operand->Data[flatIndex] == Data[i] )
            {
                operand->Gradient->Data[flatIndex] += Gradient->Data[i];
                break;
            }
        }
    }
}