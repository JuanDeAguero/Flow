// Copyright (c) 2023-2025 Juan M. G. de Agüero

#include "Flow/NArray.h"

NARRAY Flow::Matmul(NARRAY arr1, NARRAY arr2) {
    int dim1 = arr1->GetShape().size();
    int dim2 = arr2->GetShape().size();
    if (dim1 == 1 && dim2 == 1) return Sum(Mul(arr1, arr2), 0);
    else if (dim1 == 2 && dim2 == 1) return Squeeze(MM(arr1, Unsqueeze(arr2, 1)), 1);
    else if (dim1 == 1 && dim2 == 2) return Squeeze(MM(Unsqueeze(arr1, 0), arr2), 0);
    else if (dim1 == 2 && dim2 == 2) return MM(arr1, arr2);
    else if (dim1 >= 3 && (dim2 == 1 || dim2 == 2)) {
        NARRAY mm2 = arr2;
        if (dim2 == 1) mm2 = Unsqueeze(arr2, arr2->GetShape().size());
        vector<int> resultShape;
        for (int i = 0; i < arr1->GetShape().size() - 1; i++) resultShape.push_back(arr1->GetShape()[i]);
        if (dim2 > 1) resultShape.push_back(mm2->GetShape()[ mm2->GetShape().size() - 1 ]);
        vector<int> shape1;
        shape1.push_back(arr1->GetShape()[0]);
        for (int i = 1; i < arr1->GetShape().size() - 1; i++) shape1[0] *= arr1->GetShape()[i];
        shape1.push_back(arr1->GetShape()[ arr1->GetShape().size() - 1 ]);
        NARRAY mm1 = Reshape(arr1, shape1);
        return Reshape(MM(mm1, mm2), resultShape);
    } else if ((dim2 >= 1 && dim2 >= 1) && (dim2 >= 3 || dim2 >= 3)) {
        int n = 1;
        if (dim1 > 1) n = arr1->GetShape()[ arr1->GetShape().size() - 2 ];
        int m1 = arr1->GetShape()[ arr1->GetShape().size() - 1 ];
        int m2 = 1;
        if (dim2 > 1) m2 = arr2->GetShape()[ arr2->GetShape().size() - 2 ];
        int p = arr2->GetShape()[ arr2->GetShape().size() - 1 ];
        int max1 = max(dim1 - 2, 0);
        vector<int> batch1;
        for (int i = 0; i < max1; i++) batch1.push_back(arr1->GetShape()[i]);
        int max2 = max(dim2 - 2, 0);
        vector<int> batch2;
        for (int i = 0; i < max2; i++) batch2.push_back(arr2->GetShape()[i]);
        vector<int> batches = BroadcastShapes(batch1, batch2);
        vector<int> shape1, shape2;
        for (int batch : batches) { shape1.push_back(batch); shape2.push_back(batch); }
        shape1.push_back(n);
        shape1.push_back(m1);
        shape2.push_back(m2);
        shape2.push_back(p);
        int batchProduct = 1.0f;
        for (int batch : batches) batchProduct *= batch;
        vector<int> bmmShape1 = { batchProduct, n, m1 };
        vector<int> bmmShape2 = { batchProduct, m2, p };
        NARRAY arrBroadcasted1 = Reshape(Broadcast(arr1, shape1), bmmShape1);
        NARRAY arrBroadcasted2 = Reshape(Broadcast(arr2, shape2), bmmShape2);
        vector<int> resultShape = batches;
        if (dim1 > 1) resultShape.push_back(n);
        if (dim2 > 1) resultShape.push_back(p);
        NARRAY bmm = BMM(arrBroadcasted1, arrBroadcasted2);
        return Reshape(bmm, resultShape);
    }
}