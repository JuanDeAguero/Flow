// Copyright (c) 2023 Juan M. G. de Agüero

#include "Flow/NArrayCore.h"

namespace Flow
{
    NArrayCore* Add( NArrayCore* arr1, NArrayCore* arr2 )
    {
        vector<int> shape = GetShapeForBroadcast( arr1->GetShape(), arr2->GetShape() );
        NArrayCore* arr1B = Broadcast( arr1, shape );
        NArrayCore* arr2B = Broadcast( arr2, shape );

        if (UseCUDA)
            return Add_CUDA( arr1B, arr2B );

        vector<float> resultData(arr1B->Get().size());
        for ( int i = 0; i < arr1B->Get().size(); i++ )
            resultData[i] = arr1B->Get()[i] + arr2B->Get()[i];
        return new NArrayCore( arr1B->GetShape(), resultData, { arr1B, arr2B }, NArrayCore::Operation::ADD );
    }
}

void Flow::NArrayCore::BackwardAdd()
{
    if (UseCUDA)
    {
        BackwardAdd_CUDA();
        return;
    }

    for ( int i = 0; i < Gradient->Data.size(); i++ )
    {
        Operands[0]->Gradient->Data[i] += Gradient->Data[i];
        Operands[1]->Gradient->Data[i] += Gradient->Data[i];
    }
}