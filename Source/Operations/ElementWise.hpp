// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include "Flow/NArrayCore.h"

namespace Flow
{
    static void ComputeOperation( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op, vector<int> index )
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
    }

    static void ElementWise( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op )
    {
        vector<int> shape = arr1->GetShape();
        int numDims = shape.size();
        if ( numDims == 1 )
        {
            for ( int i = 0; i < shape[0]; i++ )
            {
                vector<int> index = { i };
                ComputeOperation( arr1, arr2, result, op, index );
            }
        }
        else if ( numDims == 2 )
        {
            for ( int i = 0; i < shape[0]; i++ )
            {
                for ( int j = 0; j < shape[1]; j++ )
                {
                    vector<int> index = { i, j };
                    ComputeOperation( arr1, arr2, result, op, index );
                }
            }
        }
        else if ( numDims == 3 )
        {
            for ( int i = 0; i < shape[0]; i++ )
            {
                for ( int j = 0; j < shape[1]; j++ )
                {
                    for ( int k = 0; k < shape[2]; k++ )
                    {
                        vector<int> index = { i, j, k };
                        ComputeOperation( arr1, arr2, result, op, index );
                    }
                }
            }
        }
        else if ( numDims == 4 )
        {
            for ( int i = 0; i < shape[0]; i++ )
            {
                for ( int j = 0; j < shape[1]; j++ )
                {
                    for ( int k = 0; k < shape[2]; k++ )
                    {
                        for ( int x = 0; x < shape[3]; x++ )
                        {
                            vector<int> index = { i, j, k, x };
                            ComputeOperation( arr1, arr2, result, op, index );
                        }
                    }
                }
            }
        }
        else if ( numDims == 5 )
        {
            for ( int i = 0; i < shape[0]; i++ )
            {
                for ( int j = 0; j < shape[1]; j++ )
                {
                    for ( int k = 0; k < shape[2]; k++ )
                    {
                        for ( int x = 0; x < shape[3]; x++ )
                        {
                            for ( int y = 0; y < shape[4]; y++ )
                            {
                                vector<int> index = { i, j, k, x, y };
                                ComputeOperation( arr1, arr2, result, op, index );
                            }
                        }
                    }
                }
            }
        }
    }

    void ElementWise_CUDA( NArrayCore* arr1, NArrayCore* arr2, NArrayCore* result, NArrayCore::Operation op );
}