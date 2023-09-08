// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include <torch/torch.h>

#pragma once

using namespace std;

static pair< vector<int>, vector<float> > TorchAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
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

static pair< vector<int>, vector<float> > TorchSub( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 - tensor2;
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 * tensor2;
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchMMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::mm( tensor1, tensor2 );
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchPow( pair< vector<int>, vector<float> > arr, float exponent )
{
    vector<int> arrShape = arr.first;
    vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> data = arr.second;
    torch::Tensor tensor = torch::from_blob( data.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::pow( tensor, exponent );
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchTanh( pair< vector<int>, vector<float> > arr )
{
    vector<int> arrShape = arr.first;
    vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> data = arr.second;
    torch::Tensor tensor = torch::from_blob( data.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::tanh( tensor );
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchExp( pair< vector<int>, vector<float> > arr )
{
    vector<int> arrShape = arr.first;
    vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> data = arr.second;
    torch::Tensor tensor = torch::from_blob( data.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::exp( tensor );
    vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<float>, vector<float> > TorchBackwardAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2,
                                                              pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int> arrShape3 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<int64_t> shape3( arrShape3.begin(), arrShape3.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    vector<float> data3 = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data3.data(), shape3, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 + tensor2;
    resultTensor.backward(grad);
    vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardSub( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2,
                                                              pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int> arrShape3 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<int64_t> shape3( arrShape3.begin(), arrShape3.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    vector<float> data3 = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data3.data(), shape3, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 - tensor2;
    resultTensor.backward(grad);
    vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2,
                                                               pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int> arrShape3 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<int64_t> shape3( arrShape3.begin(), arrShape3.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    vector<float> data3 = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data3.data(), shape3, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 * tensor2;
    resultTensor.backward(grad);
    vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardMMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2,
                                                                pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = arr2.first;
    vector<int> arrShape3 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<int64_t> shape3( arrShape3.begin(), arrShape3.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = arr2.second;
    vector<float> data3 = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data3.data(), shape3, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::mm( tensor1, tensor2 );
    resultTensor.backward(grad);
    vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static vector<float> TorchBackwardPow( pair< vector<int>, vector<float> > arr1, float exponent, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = gradient.second;
    torch::Tensor tensor = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::pow( tensor, exponent );
    resultTensor.backward(grad);
    vector<float> resultData( tensor.grad().data_ptr<float>(), tensor.grad().data_ptr<float>() + tensor.grad().numel() );
    return resultData;
}

static vector<float> TorchBackwardTanh( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = gradient.second;
    torch::Tensor tensor = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::tanh(tensor);
    resultTensor.backward(grad);
    vector<float> resultData( tensor.grad().data_ptr<float>(), tensor.grad().data_ptr<float>() + tensor.grad().numel() );
    return resultData;
}

static vector<float> TorchBackwardExp( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arrShape1 = arr1.first;
    vector<int> arrShape2 = gradient.first;
    vector<int64_t> shape1( arrShape1.begin(), arrShape1.end() );
    vector<int64_t> shape2( arrShape2.begin(), arrShape2.end() );
    vector<float> data1 = arr1.second;
    vector<float> data2 = gradient.second;
    torch::Tensor tensor = torch::from_blob( data1.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( data2.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::exp(tensor);
    resultTensor.backward(grad);
    vector<float> resultData( tensor.grad().data_ptr<float>(), tensor.grad().data_ptr<float>() + tensor.grad().numel() );
    return resultData;
}