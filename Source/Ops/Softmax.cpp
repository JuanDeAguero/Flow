// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Softmax(NARRAY arr, int dim) {
    NARRAY index = Create({ 1 }, { 0.0f });
    NARRAY exp = Exp(Sub(arr, Index(Max(arr, dim), dim, index)));
    return Div(exp, Sum(exp, dim));
}