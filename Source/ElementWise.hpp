// Copyright (c) 2023 Juan M. G. de Agüero

#include "Log.h"
#include "NArray.h"

#pragma once

namespace Flow
{
    NArrayCore* ElementWise( NArrayCore* arr1, NArrayCore* arr2, NArrayCore::Operation op )
    {
        if ( arr1->GetShape().size() > 2 ||  arr2->GetShape().size() > 2 )
        {
            Log("[Error] Only 1D and 2D arrays are supported for addition.");
            return nullptr;
        }

        // Create a copy of the two arrays.
        // They might need to be reshaped and we don't want to modify the input arrays.
        NArrayCore* arr1Copy = new NArrayCore( arr1->GetShape(), arr1->Get() );
        NArrayCore* arr2Copy = new NArrayCore( arr2->GetShape(), arr2->Get() );

        // Add 1s if needed.
        if ( arr1->GetShape().size() == 2 && arr2->GetShape().size() == 1 )
            arr2Copy->Reshape({ 1, arr2->GetShape()[0] });
        else if ( arr1->GetShape().size() == 1 && arr2->GetShape().size() == 2 )
            arr1Copy->Reshape({ 1, arr1->GetShape()[0] });

        // Check if shapes are compatible.
        for ( int i = 0; i < arr1Copy->GetShape().size(); i++ )
        {
            if ( arr1Copy->GetShape()[i] != arr2Copy->GetShape()[i]
                && arr1Copy->GetShape()[i] != 1 && arr2Copy->GetShape()[i] != 1 )
            {
                Log("[Error] Array shapes are incompatible for addition.");
                return nullptr;
            }
        }

        // Create the result array.
        vector<int> resultShape;
        for ( int i = 0; i < arr1Copy->GetShape().size(); i++ )
        {
            if ( arr1Copy->GetShape()[i] != 1 )
                resultShape.push_back(arr1Copy->GetShape()[i]);
            else resultShape.push_back(arr2Copy->GetShape()[i]);
        }
        int resultSize = resultShape[0];
        for ( int i = 1; i < resultShape.size(); i++ )
            resultSize *= resultShape[i];
        vector<float> resultData( resultSize, 0.0f );
        NArrayCore* result = new NArrayCore( resultShape, resultData, { arr1, arr2 }, op );
        
        vector<int> coords1;
        vector<int> coords2;

        // 1D
        for ( int i = 0; i < resultShape[0]; i++ )
        {
            if ( arr1Copy->GetShape().size() == 1 )
            {
                switch (op)
                {
                    case NArrayCore::Operation::ADD:
                        result->Set( { i }, arr1Copy->Get({ i }) + arr2Copy->Get({ i }) );
                        break;
                    case NArrayCore::Operation::SUB:
                        result->Set( { i }, arr1Copy->Get({ i }) - arr2Copy->Get({ i }) );
                        break;
                    case NArrayCore::Operation::MULT:
                        result->Set( { i }, arr1Copy->Get({ i }) * arr2Copy->Get({ i }) );
                        break;
                    case NArrayCore::Operation::DIV:
                        result->Set( { i }, arr1Copy->Get({ i }) / arr2Copy->Get({ i }) );
                        break;
                }
            }
            else
            {
                // 2D
                for ( int j = 0; j < resultShape[1]; j++ )
                {
                    coords1 = { i, j };
                    coords2 = { i, j };
                    if ( arr1Copy->GetShape()[0] == 1 ) coords1[0] = 0;
                    if ( arr1Copy->GetShape()[1] == 1 ) coords1[1] = 0;
                    if ( arr2Copy->GetShape()[0] == 1 ) coords2[0] = 0;
                    if ( arr2Copy->GetShape()[1] == 1 ) coords2[1] = 0;
                    if ( arr1Copy->GetShape().size() == 2 )
                    {
                        switch (op)
                        {
                            case NArrayCore::Operation::ADD:
                                result->Set( { i, j }, arr1Copy->Get(coords1) + arr2Copy->Get(coords2) );
                                break;
                            case NArrayCore::Operation::SUB:
                                result->Set( { i, j }, arr1Copy->Get(coords1) - arr2Copy->Get(coords2) );
                                break;
                            case NArrayCore::Operation::MULT:
                                result->Set( { i, j }, arr1Copy->Get(coords1) * arr2Copy->Get(coords2) );
                                break;
                            case NArrayCore::Operation::DIV:
                                result->Set( { i, j }, arr1Copy->Get(coords1) / arr2Copy->Get(coords2) );
                                break;
                        }
                    }
                    else
                    {
                        // 3D ...
                    }
                }
            }
        }

        return result;
    }
}