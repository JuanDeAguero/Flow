// Copyright (c) 2023 Juan M. G. de Agüero

#include "Log.h"
#include "NArray.h"

#pragma once

namespace Flow
{
    /**
     * @brief Provides an element-wise operation functionality on NArrays.
     *
     * @details The "ElementWise" function namespace performs element-wise operations 
     * (addition, subtraction, multiplication, ...) on NArray objects. This implementation 
     * supports operations on both 1D and 2D arrays.
     *
     * Shape verification:
     * The function checks if the arrays are either 1D or 2D. It doesn't support arrays 
     * with dimensions higher than 2.
     *
     * Reshaping for broadcasting:
     * The function can reshape 1D arrays to align with 2D arrays for operations, enabling 
     * broadcasting functionality similar to what is found in numpy.
     * 
     * Shape compatibility:
     * Before performing operations, the function ensures that the two arrays have 
     * compatible shapes. Two shapes are considered compatible if they are equal, or one 
     * of them is a 1D shape that can be broadcasted to match the 2D shape of the other.
     *
     * Performing element-wise operation:
     * The operation to be performed (addition, subtraction, multiplication, or division) is 
     * determined by the `NArray::Operation` enumeration passed to the function. The operation 
     * is applied element-wise, taking into account the broadcasting rules if necessary.
     *
     * @param arr1 Pointer to the first NArray object.
     * @param arr2 Pointer to the second NArray object.
     * @param op The element-wise operation to perform from NArray::Operation.
     * 
     * @return Pointer to the resulting NArray after performing the specified operation or 
     * nullptr if the operation couldn't be performed due to shape incompatibility.
     */
    NArray* ElementWise( NArray* arr1, NArray* arr2, NArray::Operation op )
    {
        if ( arr1->GetShape().size() > 2 ||  arr2->GetShape().size() > 2 )
        {
            Log("[Error] Only 1D and 2D arrays are supported for addition.");
            return nullptr;
        }

        // Create a copy of the two arrays.
        // They might need to be reshaped and we don't want to modify the input arrays.
        NArray* arr1Copy = new NArray( arr1->GetShape(), arr1->Get(), true );
        NArray* arr2Copy = new NArray( arr2->GetShape(), arr2->Get(), true );

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
        NArray* result = new NArray( resultShape, resultData, { arr1, arr2 }, op );

        // The two arrays have compatible shapes so we can add them.
        vector<int> coords1;
        vector<int> coords2;
        // 1D
        for ( int i = 0; i < resultShape[0]; i++ )
        {
            if ( arr1Copy->GetShape().size() == 1 )
            {
                switch (op)
                {
                    case NArray::Operation::ADD:
                        result->Set( { i }, arr1Copy->Get({ i }) + arr2Copy->Get({ i }) );
                        break;
                    case NArray::Operation::SUB:
                        result->Set( { i }, arr1Copy->Get({ i }) - arr2Copy->Get({ i }) );
                        break;
                    case NArray::Operation::MULT:
                        result->Set( { i }, arr1Copy->Get({ i }) * arr2Copy->Get({ i }) );
                        break;
                    case NArray::Operation::DIV:
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
                            case NArray::Operation::ADD:
                                result->Set( { i, j }, arr1Copy->Get(coords1) + arr2Copy->Get(coords2) );
                                break;
                            case NArray::Operation::SUB:
                                result->Set( { i, j }, arr1Copy->Get(coords1) - arr2Copy->Get(coords2) );
                                break;
                            case NArray::Operation::MULT:
                                result->Set( { i, j }, arr1Copy->Get(coords1) * arr2Copy->Get(coords2) );
                                break;
                            case NArray::Operation::DIV:
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