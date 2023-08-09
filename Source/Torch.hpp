// Copyright (c) 2023 Juan M. G. de Agüero

#include <vector>

#include <torch/torch.h>

using namespace std;

static pair< vector<int>, vector<float> > TorchAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 + tensor2;
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchSub( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 - tensor2;
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 * tensor2;
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchMMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2 )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::mm( tensor1, tensor2 );
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchPow( pair< vector<int>, vector<float> > arr, float exponent )
{
    vector<int> arrShape = arr.first;
    std::vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> arrData = arr.second;
    torch::Tensor tensor = torch::from_blob( arrData.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::pow( tensor, exponent );
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchTanh( pair< vector<int>, vector<float> > arr )
{
    vector<int> arrShape = arr.first;
    std::vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> arrData = arr.second;
    torch::Tensor tensor = torch::from_blob( arrData.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::tanh( tensor );
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<int>, vector<float> > TorchExp( pair< vector<int>, vector<float> > arr )
{
    vector<int> arrShape = arr.first;
    std::vector<int64_t> shape( arrShape.begin(), arrShape.end() );
    vector<float> arrData = arr.second;
    torch::Tensor tensor = torch::from_blob( arrData.data(), shape, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::exp( tensor );
    std::vector<int> resultShape( resultTensor.sizes().begin(), resultTensor.sizes().end() );
    std::vector<float> resultData( resultTensor.data_ptr<float>(), resultTensor.data_ptr<float>() + resultTensor.numel() );
    return { resultShape, resultData };
}

static pair< vector<float>, vector<float> > TorchBackwardAdd( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 + tensor2;
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    std::vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardSub( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 - tensor2;
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    std::vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = tensor1 * tensor2;
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    std::vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static pair< vector<float>, vector<float> > TorchBackwardMMult( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > arr2, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> arr2Shape = arr2.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shape2( arr2Shape.begin(), arr2Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> arr2Data = arr2.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor tensor2 = torch::from_blob( arr2Data.data(), shape2, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::mm( tensor1, tensor2 );
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    std::vector<float> resultData2( tensor2.grad().data_ptr<float>(), tensor2.grad().data_ptr<float>() + tensor2.grad().numel() );
    return { resultData1, resultData2 };
}

static vector<float> TorchBackwardPow( pair< vector<int>, vector<float> > arr1, float exponent, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::pow( tensor1, exponent );
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    return resultData1;
}

static vector<float> TorchBackwardTanh( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::tanh( tensor1 );
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    return resultData1;
}

static vector<float> TorchBackwardExp( pair< vector<int>, vector<float> > arr1, pair< vector<int>, vector<float> > gradient )
{
    vector<int> arr1Shape = arr1.first;
    vector<int> gradShape = gradient.first;
    std::vector<int64_t> shape1( arr1Shape.begin(), arr1Shape.end() );
    std::vector<int64_t> shapeGrad( gradShape.begin(), gradShape.end() );
    vector<float> arr1Data = arr1.second;
    vector<float> gradData = gradient.second;
    torch::Tensor tensor1 = torch::from_blob( arr1Data.data(), shape1, torch::dtype(torch::kFloat32).requires_grad(true) );
    torch::Tensor grad = torch::from_blob( gradData.data(), shapeGrad, torch::dtype(torch::kFloat32) );
    torch::Tensor resultTensor = torch::exp( tensor1 );
    resultTensor.backward(grad);
    std::vector<float> resultData1( tensor1.grad().data_ptr<float>(), tensor1.grad().data_ptr<float>() + tensor1.grad().numel() );
    return resultData1;
}