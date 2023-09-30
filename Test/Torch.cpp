// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include <Flow/Print.h>

#include <torch/torch.h>

using namespace std;

namespace Flow
{
    void Print( pair< vector<int>, vector<float> > arr )
    {
        vector<int> shape = arr.first;
        vector<float> data = arr.second;
        Print("Shape:");
        for ( int value : shape ) Print(static_cast<float>(value));
        Print("Data:");
        for ( float value : data ) Print(value);
    }
    
    pair< vector<int>, vector<float> > TorchAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
    {
        vector<int> arrShape1 = arr1.first;
        vector<int> arrShape2 = arr2.first;
        vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
        vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
        vector<float> data1 = arr1.second;
        vector<float> data2 = arr2.second;
        torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32) );
        torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
        torch::Tensor resultTensor = tensor1 + tensor2;
        vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
        vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
        return { resultShape, resultData };
    }
}