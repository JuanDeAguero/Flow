// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 )
    {
        if ( arr1->GetShape().size() != 2 || arr2->GetShape().size() != 2 )
            throw runtime_error("MM only supports 2D x 2D.");

        if (UseCUDA)
            return MM_CUDA( arr1, arr2 );

        int arr1Rows = arr1->GetShape()[0];
        int arr1Cols = arr1->GetShape()[1];
        int arr2Cols = arr2->GetShape()[1];
        vector<float> resultData( arr1Rows * arr2Cols, 0.0f );
        for ( int i = 0; i < arr1Rows; i++ )
        {
            for ( int j = 0; j < arr2Cols; j++ )
            {
                float sum = 0.0f;
                for ( int k = 0; k < arr1Cols; k++ )
                    sum += arr1->Get({ i, k }) * arr2->Get({ k, j });
                resultData[ i * arr2Cols + j ] = sum;
            }
        }
        return new NArrayCore( { arr1Rows, arr2Cols }, resultData, { arr1, arr2 }, NArrayCore::Operation::MM );
    }
}

void Flow::NArrayCore::BackwardMM()
{
    NArrayCore* grad1 = Flow::MM( Gradient, Transpose( Operands[1], 0, 1 ) );
    NArrayCore* grad2 = Flow::MM( Transpose( Operands[0], 0, 1 ), Gradient );
    Operands[0]->Gradient = grad1;
    Operands[1]->Gradient = grad2;
    grad1->Gradient = nullptr;
    grad2->Gradient = nullptr;
}