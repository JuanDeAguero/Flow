// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArrayCore.h"

namespace Flow
{
    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op );

    static void ElementWise( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        if (UseCUDA)
        {
            ElementWise_CUDA( arr1, arr2, result, op );
            return;
        }
        
        for ( int i = 0; i < arr1->Get().size(); i++ )
        {
            vector<int> index = FlatToMultiIndex( i, arr1->GetShape() );
            switch (op)
            {
                case NArrayCore::Operation::ADD:
                    result->Set( index, arr1->Get(index) + arr2->Get(index) );
                    break;
                case NArrayCore::Operation::MUL:
                    result->Set( index, arr1->Get(index) * arr2->Get(index) );
                    break;
            }
        }
    }
}