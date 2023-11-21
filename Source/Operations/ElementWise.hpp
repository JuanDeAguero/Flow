// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <chrono>
#include <string>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op );

    static void ElementWise( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        if (UseCUDA)
        {
            auto start = chrono::high_resolution_clock::now();
            ElementWise_CUDA( arr1, arr2, result, op );
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>( end - start );
            //Print( to_string(duration.count()) );
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