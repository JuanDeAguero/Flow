// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArrayCore.h"

#ifndef FLOW_CUDA

namespace Flow
{
    static void ElementWise( vector<int>& index, NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        if ( index.size() == arr1->GetShape().size() )
        {
            switch (op)
            {
                case NArrayCore::Operation::ADD:
                    result->Set( index, arr1->Get(index) + arr2->Get(index) );
                    break;
                case NArrayCore::Operation::MUL:
                    result->Set( index, arr1->Get(index) * arr2->Get(index) );
                    break;
            }
            return;
        }
        for ( int i = 0; i < arr1->GetShape()[index.size()]; i++ )
        {
            vector<int> newIndex = index;
            newIndex.push_back(i);
            ElementWise( newIndex, arr1, arr2, result, op );
        }
    }
}

#else

namespace Flow
{
    static void ElementWise( vector<int>& index, NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        
    }
}

#endif