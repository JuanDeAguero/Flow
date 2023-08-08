// Copyright (c) Juan M. G. de Agüero 2023

#include <vector>

#include <torch/torch.h>

using namespace std;

static pair< vector<int>, vector<float> > TorchMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1 );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2 );
    torch::Tensor resultTensor = tensor1 * tensor2;
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}