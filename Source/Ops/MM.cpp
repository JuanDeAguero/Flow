// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::MM(NARRAY arr1, NARRAY arr2) {
    return Squeeze(BMM(Unsqueeze(arr1, 0), Unsqueeze(arr2, 0)), 0);
}