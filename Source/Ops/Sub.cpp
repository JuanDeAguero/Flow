// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Sub(NARRAY arr1, NARRAY arr2) {
    return Add(arr1, Neg(arr2));
}