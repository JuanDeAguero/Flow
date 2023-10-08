// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 )
    {
        if ( arr1->GetShape().size() != 2 || arr2->GetShape().size() != 2 )
            throw runtime_error("[MM] Both arrays must be 2D.");
        if ( arr1->GetShape()[1] != arr2->GetShape()[0] )
            throw runtime_error("[MM] Inner dimensions do not match.");
        int m = arr1->GetShape()[0];
        int n = arr1->GetShape()[1];
        int p = arr2->GetShape()[1];
        vector<float> resultData( m * p, 0.0f );
        for ( int i = 0; i < m; i++ )
        {
            for ( int j = 0; j < p; j++ )
            {
                float sum = 0.0f;
                for ( int k = 0; k < n; k++ )
                    sum += arr1->Get({ i, k }) * arr2->Get({ k, j });
                resultData[ i * p + j ] = sum;
            }
        }
        return new NArrayCore( { m, p }, resultData, { arr1, arr2 }, NArrayCore::Operation::MM );
    }
}

void Flow::NArrayCore::BackwardMM()
{
    NArrayCore* operand1 = Operands[0];
    NArrayCore* operand2 = Operands[1];
    int m = operand1->GetShape()[0];
    int n = operand1->GetShape()[1];
    int p = operand2->GetShape()[1];
    for ( int i = 0; i < m; i++ )
    {
        for ( int j = 0; j < n; j++ )
        {
            float sum = 0.0f;
            for ( int k = 0; k < p; k++ )
                sum += Gradient->Get({ i, k }) * operand2->Get({ j, k });
            operand1->Gradient->Data[ i * n + j ] += sum;
        }
    }
    for ( int i = 0; i < n; i++ )
    {
        for ( int j = 0; j < p; j++ )
        {
            float sum = 0.0f;
            for ( int k = 0; k < m; k++ )
                sum += operand1->Get({ k, i }) * Gradient->Get({ k, j });
            operand2->Gradient->Data[ i * p + j ] += sum;
        }
    }
}