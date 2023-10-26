// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 )
    {
        if (UseCUDA)
            return MM_CUDA( arr1, arr2 );

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
    if (UseCUDA)
    {
        BackwardMM_CUDA();
        return;
    }

    int m = Operands[0]->GetShape()[0];
    int n = Operands[0]->GetShape()[1];
    int p = Operands[1]->GetShape()[1];
    for ( int i = 0; i < m; i++ )
    {
        for ( int j = 0; j < n; j++ )
        {
            float sum = 0.0f;
            for ( int k = 0; k < p; k++ )
                sum += Gradient->Get({ i, k }) * Operands[1]->Get({ j, k });
            Operands[0]->Gradient->Data[ i * n + j ] += sum;
        }
    }
    for ( int i = 0; i < n; i++ )
    {
        for ( int j = 0; j < p; j++ )
        {
            float sum = 0.0f;
            for ( int k = 0; k < m; k++ )
                sum += Operands[0]->Get({ k, i }) * Gradient->Get({ k, j });
            Operands[1]->Gradient->Data[ i * p + j ] += sum;
        }
    }
}