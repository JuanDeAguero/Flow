// Copyright (c) 2023-2024 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Neg( NARRAY arr )
{
    return Mul( arr, -1.0f );
}