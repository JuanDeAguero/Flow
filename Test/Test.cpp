// Copyright (c) Juan M. G. de Agüero 2023

#include <chrono>

#include <torch/torch.h>

#include "Flow.h"
#include "Test.h"

int main()
{
    NumTests = 1; NumPassed = 0;
    TestAdd( { 3 }, { 3, 3 } );
    Flow::Print( "Test_Add " + to_string(NumPassed) + "/" + to_string(NumTests) );

    NumTests = 8; NumPassed = 0;
    TestBroadcast( { 3 }, { 3, 3 } );
    TestBroadcast( { 1, 4, 10 }, { 3, 5, 4, 10 } );
    TestBroadcast( { 4, 4, 5 }, { 4, 4, 5 } );
    TestBroadcast( { 3, 3 }, { 1, 3, 3 } );
    TestBroadcast( { 1 }, { 3, 3, 3 } );
    TestBroadcast( { 1, 1, 1, 3 }, { 4, 4, 4, 3 } );
    TestBroadcast( { 2, 1, 3 }, { 2, 5, 3 } );
    TestBroadcast( { 3, 1, 1, 2, 1, 3 }, { 2, 3, 10, 5, 2, 5, 3 } );
    Flow::Print( "Test_Broadcast " + to_string(NumPassed) + "/" + to_string(NumTests) );

    NumTests = 1; NumPassed = 0;
    TestExp({ 3, 3 });
    Flow::Print( "Test_Exp " + to_string(NumPassed) + "/" + to_string(NumTests) );

    NumTests = 1; NumPassed = 0;
    TestSum( { 3, 3 }, 0 );
    Flow::Print( "Test_Sum " + to_string(NumPassed) + "/" + to_string(NumTests) );
}

void TestAdd( vector<int> arrShape1, vector<int> arrShape2 )
{
    NARRAY arr1 = Flow::RandomUniform( arrShape1, -0.9999f, 0.9999f );
    NARRAY arr2 = Flow::RandomUniform( arrShape2, -0.9999f, 0.9999f );
    NARRAY result = Flow::Add( arr1, arr2 );
    result->Backpropagate();
    NARRAY grad1 = arr1->GetGradient();
    NARRAY grad2 = arr2->GetGradient();
    vector<int64_t> arr1Shape64 = Flow::ToInt64(arrShape1);
    torch::IntArrayRef shapeRef1(arr1Shape64);
    vector<int64_t> arr2Shape64 = Flow::ToInt64(arrShape2);
    torch::IntArrayRef shapeRef2(arr2Shape64);
    torch::Tensor tensor1 = torch::tensor( arr1->Get(), torch::requires_grad(true) ).to(torch::kCUDA);
    tensor1 = tensor1.reshape(shapeRef1);
    tensor1.retain_grad();
    torch::Tensor tensor2 = torch::tensor( arr2->Get(), torch::requires_grad(true) ).to(torch::kCUDA);
    tensor2 = tensor2.reshape(shapeRef2);
    tensor2.retain_grad();
    auto resultTorch = tensor1.add(tensor2);
    resultTorch.backward(torch::ones_like(resultTorch).to(torch::kCUDA));
    torch::cuda::synchronize();
    auto gradTorch1 = tensor1.grad().contiguous();
    auto gradTorch2 = tensor2.grad().contiguous();
    resultTorch = resultTorch.contiguous();
    auto resultTorchCpu = resultTorch.cpu();
    auto gradTorchCpu1 = gradTorch1.cpu();
    auto gradTorchCpu2 = gradTorch2.cpu();
    float* resultTorchPtr = resultTorchCpu.data_ptr<float>();
    float* gradTorchPtr1 = gradTorchCpu1.data_ptr<float>();
    float* gradTorchPtr2 = gradTorchCpu2.data_ptr<float>();
    vector<float> resultTorchData( resultTorchPtr, resultTorchPtr + resultTorchCpu.numel() );
    vector<float> gradTorchData1( gradTorchPtr1, gradTorchPtr1 + gradTorchCpu1.numel() );
    vector<float> gradTorchData2( gradTorchPtr2, gradTorchPtr2 + gradTorchCpu2.numel() );
    bool dataCorrect = Flow::Equals( result->Get(), resultTorchData, 0.000001f );
    bool gradCorrect1 = Flow::Equals( grad1->Get(), gradTorchData1, 0.000001f );
    bool gradCorrect2 = Flow::Equals( grad2->Get(), gradTorchData2, 0.000001f );
    if ( dataCorrect && gradCorrect1 && gradCorrect2 ) NumPassed++;
}

void TestBroadcast( vector<int> arrShape, vector<int> broadcastShape )
{
    NARRAY arr = Flow::RandomUniform( arrShape, -0.9999f, 0.9999f );
    NARRAY result = Flow::Broadcast( arr, broadcastShape );
    //auto start = chrono::high_resolution_clock::now();
    result->Backpropagate();
    //auto end = chrono::high_resolution_clock::now();
    //auto duration = chrono::duration_cast<chrono::microseconds>( end - start ).count();
    //Flow::Print(duration);
    NARRAY grad = arr->GetGradient();
    vector<int64_t> shape64 = Flow::ToInt64(arrShape);
    torch::IntArrayRef shapeRef(shape64);
    vector<int64_t> broadcastShape64 = Flow::ToInt64(broadcastShape);
    torch::IntArrayRef broadcastShapeRef(broadcastShape64);
    torch::Tensor tensor = torch::tensor( arr->Get(), torch::requires_grad(true) ).to(torch::kCUDA);
    tensor = tensor.reshape(shapeRef);
    tensor.retain_grad();
    auto resultTorch = tensor.expand(broadcastShapeRef);
    //start = chrono::high_resolution_clock::now();
    resultTorch.backward(torch::ones_like(resultTorch).to(torch::kCUDA));
    torch::cuda::synchronize();
    //end = chrono::high_resolution_clock::now();
    //duration = chrono::duration_cast<chrono::microseconds>( end - start ).count();
    //Flow::Print(duration);
    auto gradTorch = tensor.grad().contiguous();
    resultTorch = resultTorch.contiguous();
    auto resultTorchCpu = resultTorch.cpu();
    auto gradTorchCpu = gradTorch.cpu();
    float* resultTorchPtr = resultTorchCpu.data_ptr<float>();
    float* gradTorchPtr = gradTorchCpu.data_ptr<float>();
    vector<float> resultTorchData( resultTorchPtr, resultTorchPtr + resultTorchCpu.numel() );
    vector<float> gradTorchData( gradTorchPtr, gradTorchPtr + gradTorchCpu.numel() );
    bool dataCorrect = Flow::Equals( result->Get(), resultTorchData, 0.000001f );
    bool gradCorrect = Flow::Equals( grad->Get(), gradTorchData, 0.000001f );
    if ( dataCorrect && gradCorrect ) NumPassed++;
}

void TestExp( vector<int> arrShape )
{
    NARRAY arr = Flow::RandomUniform( arrShape, -0.9999f, 0.9999f );
    NARRAY result = Flow::Exp(arr);
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
    vector<int64_t> shape64 = Flow::ToInt64(arrShape);
    torch::IntArrayRef shapeRef(shape64);
    torch::Tensor tensor = torch::tensor( arr->Get(), torch::requires_grad(true) ).to(torch::kCUDA);
    tensor = tensor.reshape(shapeRef);
    tensor.retain_grad();
    auto resultTorch = tensor.exp();
    resultTorch.backward(torch::ones_like(resultTorch).to(torch::kCUDA));
    torch::cuda::synchronize();
    auto gradTorch = tensor.grad().contiguous();
    resultTorch = resultTorch.contiguous();
    auto resultTorchCpu = resultTorch.cpu();
    auto gradTorchCpu = gradTorch.cpu();
    float* resultTorchPtr = resultTorchCpu.data_ptr<float>();
    float* gradTorchPtr = gradTorchCpu.data_ptr<float>();
    vector<float> resultTorchData( resultTorchPtr, resultTorchPtr + resultTorchCpu.numel() );
    vector<float> gradTorchData( gradTorchPtr, gradTorchPtr + gradTorchCpu.numel() );
    bool dataCorrect = Flow::Equals( result->Get(), resultTorchData, 0.000001f );
    bool gradCorrect = Flow::Equals( grad->Get(), gradTorchData, 0.000001f );
    if ( dataCorrect && gradCorrect ) NumPassed++;
}

void TestSum( vector<int> arrShape, int dim )
{
    NARRAY arr = Flow::RandomUniform( arrShape, -0.9999f, 0.9999f );
    NARRAY result = Flow::Sum( arr, dim );
    result->Backpropagate();
    NARRAY grad = arr->GetGradient();
    vector<int64_t> shape64 = Flow::ToInt64(arrShape);
    torch::IntArrayRef shapeRef(shape64);
    torch::Tensor tensor = torch::tensor( arr->Get(), torch::requires_grad(true) ).to(torch::kCUDA);
    tensor = tensor.reshape(shapeRef);
    tensor.retain_grad();
    auto resultTorch = tensor.sum(dim);
    resultTorch.backward(torch::ones_like(resultTorch).to(torch::kCUDA));
    torch::cuda::synchronize();
    auto gradTorch = tensor.grad().contiguous();
    resultTorch = resultTorch.contiguous();
    auto resultTorchCpu = resultTorch.cpu();
    auto gradTorchCpu = gradTorch.cpu();
    float* resultTorchPtr = resultTorchCpu.data_ptr<float>();
    float* gradTorchPtr = gradTorchCpu.data_ptr<float>();
    vector<float> resultTorchData( resultTorchPtr, resultTorchPtr + resultTorchCpu.numel() );
    vector<float> gradTorchData( gradTorchPtr, gradTorchPtr + gradTorchCpu.numel() );
    bool dataCorrect = Flow::Equals( result->Get(), resultTorchData, 0.000001f );
    bool gradCorrect = Flow::Equals( grad->Get(), gradTorchData, 0.000001f );
    if ( dataCorrect && gradCorrect ) NumPassed++;
}