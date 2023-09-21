// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    NArrayCore* MM( NArrayCore* arr1, NArrayCore* arr2 )
    {
        if ( arr1->GetShape().size() != 2 || arr2->GetShape().size() != 2 )
        {
            Print("[Error] Both arrays must be 2D for matrix multiplication.");
            return nullptr;
        } 
        if ( arr1->GetShape()[1] != arr2->GetShape()[0] )
        {
            Print("[Error] Inner dimensions do not match. Matrix multiplication is not possible.");
            return nullptr;
        }
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
    NArrayCore* A = Operands[0];
    NArrayCore* B = Operands[1];
    int m = A->GetShape()[0];
    int n = A->GetShape()[1];
    int p = B->GetShape()[1];
    for (int i = 0; i < m; i++)
    {
        for (int k = 0; k < n; k++)
        {
            float gradSum = 0.0f;
            for (int j = 0; j < p; j++)
            {
                gradSum += Gradient->Get({ i, j }) * B->Get({ k, j });
            }
            A->Gradient->Data[i * n + k] += gradSum;
        }
    }
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < p; j++)
        {
            float gradSum = 0.0f;
            for (int i = 0; i < m; i++)
            {
                gradSum += A->Get({ i, k }) * Gradient->Get({ i, j });
            }
            B->Gradient->Data[k * p + j] += gradSum;
        }
    }
}