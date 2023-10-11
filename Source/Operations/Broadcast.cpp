// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

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
        }
        return shape;
    }

    NArrayCore* Broadcast( NArrayCore* arr, vector<int> shape )
    {
        if (UseCUDA)
            return Broadcast_CUDA( arr, shape );

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
    vector<float> operandGradient( Operands[0]->Data.size(), 0.0f );
    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        vector<int> position = FlatToMultiIndex( i, Shape );
        vector<int> operandCoords;
        for ( int j = 0; j < Operands[0]->Shape.size(); j++ )
        {
            int coord = position[ Shape.size() - Operands[0]->Shape.size() + j ];
            if ( Operands[0]->Shape[j] == 1 ) coord = 0;
            operandCoords.push_back(coord);
        }
        int operandIndex = MultiToFlatIndex( operandCoords, Operands[0]->Shape );
        operandGradient[operandIndex] += Gradient->Data[i];
    }
    for ( int i = 0; i < Operands[0]->Gradient->Data.size(); i++ )
        Operands[0]->Gradient->Data[i] += operandGradient[i];
}