// Copyright (c) 2023 Juan M. G. de Agüero

#include <stdexcept>

#include "Flow/NArrayCore.h"
#include "Flow/Print.h"

namespace Flow
{
    vector<int> GetShapeForBroadcast( NArrayCore* arr1, NArrayCore* arr2 )
    {
        vector<int> shape1 = arr1->GetShape();
        vector<int> shape2 = arr2->GetShape();
        int maxDims = max( shape1.size(), shape2.size() );
        while ( shape1.size() < maxDims ) shape1.insert( shape1.begin(), 1 );
        while ( shape2.size() < maxDims ) shape2.insert( shape2.begin(), 1 );
        vector<int> shape(maxDims);
        for ( int i = 0; i < maxDims; i++ )
        {
            if ( shape1[i] == shape2[i] ) shape[i] = shape1[i];
            else if ( shape1[i] == 1 ) shape[i] = shape2[i];
            else if ( shape2[i] == 1 ) shape[i] = shape1[i];
            else throw runtime_error("[GetShapeForBroadcast] The arrays are not compatible.");
        }
        return shape;
    }

    NArrayCore* Flow::Broadcast( NArrayCore* arr, vector<int> shape )
    {
        if ( shape.size() < arr->GetShape().size() )
            throw runtime_error("[Broadcast] Incompatible shape.");
        for ( int i = 1; i <= arr->GetShape().size(); i++ )
        {
            if ( shape[ shape.size() - i ] != arr->GetShape()[ arr->GetShape().size() - i ] &&
                arr->GetShape()[ arr->GetShape().size() - i ] != 1 &&
                shape[ shape.size() - i ] != 1 )
                throw runtime_error("[Broadcast] Incompatible shape.");
        }
        vector<float> resultData( SizeFromShape(shape), 0.0f );
        for ( int i = 0; i < SizeFromShape(shape); i++ )
        {
            vector<int> position = FlatToMultiIndex( i, shape );
            vector<int> originalCoords;
            for ( int j = 0; j < arr->GetShape().size(); j++ )
            {
                int coord = position[ shape.size() - arr->GetShape().size() + j ];
                if ( arr->GetShape()[j] == 1 ) coord = 0;
                originalCoords.push_back(coord);
            }
            int flatIndex = MultiToFlatIndex( originalCoords, arr->GetShape() );
            resultData[i] = arr->Get()[flatIndex];
        }
        return new NArrayCore( shape, resultData, { arr }, NArrayCore::Operation::BROADCAST );
    }
}

void Flow::NArrayCore::BackwardBroadcast()
{
    if ( Operands.size() != 1 )
        throw runtime_error("[BackwardBroadcast] Invalid number of operands.");
    NArrayCore* operand = Operands[0];
    vector<float> operandGradient( operand->Data.size(), 0.0f );
    vector<int> operandShape = operand->Shape;
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> position = FlatToMultiIndex( i, Shape );
        vector<int> operandCoords;
        for ( int j = 0; j < operandShape.size(); j++ )
        {
            int coord = position[ Shape.size() - operandShape.size() + j ];
            if ( operandShape[j] == 1 ) coord = 0;
            operandCoords.push_back(coord);
        }
        int operandIndex = MultiToFlatIndex( operandCoords, operandShape );
        operandGradient[operandIndex] += Gradient->Data[i];
    }
    for ( int i = 0; i < operand->Gradient->Data.size(); i++ )
        operand->Gradient->Data[i] += operandGradient[i];
}