// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Transpose(NARRAY arr, int firstDim, int secondDim) {
    vector<int> resultShape = arr->Shape;
    vector<int> resultStride = arr->Stride;
    swap(resultShape[firstDim], resultShape[secondDim]);
    swap(resultStride[firstDim], resultStride[secondDim]);
    NARRAY result = make_shared<NArray>(resultShape, resultStride, arr->GetOffset(), FindMetaParent(arr), vector<NARRAY>{arr}, NArray::Operation::TRANSPOSE);
    result->TransposeFirstDim = firstDim;
    result->TransposeSecondDim = secondDim;
    return result;
}

void Flow::NArray::BackwardTranspose() {
    Operands[0]->Gradient = Transpose(Gradient->Copy(), TransposeFirstDim, TransposeSecondDim);
}