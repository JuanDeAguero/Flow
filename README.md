# Flow 🌊
Machine Learning Library in C++
## Minimum requirements
- C++ 11
- CMake 3.12
- CUDA Toolkit 12.3
- Windows 10
## Features ✨
- N dimensional array operations
  - Addition, multiplication, ReLU, matrix multiplication, ...
- Autograd system
- GPU acceleration with CUDA
- Deep neural networks
## Example: MNIST classifier 🔢
See "Showcase/MNIST/".<br>
```cpp
vector<float> trainImages = ReadImagesMNIST("...");
vector<float> trainLabels = ReadLabelsMNIST("...");
vector<float> testImages = ReadImagesMNIST("...");
vector<float> testLabels = ReadLabelsMNIST("...");

int n = 6000;

trainImages.resize( 784 * n );
testImages.resize( 784 * n );
trainLabels.resize(n);
testLabels.resize(n);

NARRAY xTrain( Flow::NArray::Create( { n, 784 }, trainImages ) );
NARRAY xTest( Flow::NArray::Create( { n, 784 }, testImages ) );
NARRAY yTrain( Flow::NArray::Create( { n }, trainLabels ) );
NARRAY yTest( Flow::NArray::Create( { n }, testLabels ) );

xTrain = Flow::Div( xTrain->Copy(), Flow::NArray::Create( { 1 }, { 255.0f } ) );
xTest = Flow::Div( xTest->Copy(), Flow::NArray::Create( { 1 }, { 255.0f } ) );

NARRAY w1( Flow::Random({ 784, 512 }) );
NARRAY b1( Flow::Random({ 512 }) );
NARRAY w2( Flow::Random({ 512, 256 }) );
NARRAY b2( Flow::Random({ 256 }) );
NARRAY w3( Flow::Random({ 256, 10 }) );
NARRAY b3( Flow::Random({ 10 }) );

Flow::Optimizer optimizer( { w1, b1, w2, b2, w3, b3 }, 0.0025f, 1e-8f, 0.01f );

for ( int epoch = 0; epoch < 500; epoch++ )
{
    NARRAY a1 = ReLU( Add( MM( xTrain, w1 ), b1 ) );
    NARRAY a2 = ReLU( Add( MM( a1, w2 ), b2 ) );
    NARRAY yPredicted = Add( MM( a2, w3 ), b3 );
    NARRAY loss = CrossEntropy( yPredicted, yTrain );
    optimizer.ZeroGrad();
    loss->Backpropagate();
    optimizer.Step();
}
```
