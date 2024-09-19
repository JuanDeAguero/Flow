# Flow 🌊
Machine Learning Library in C++
## Features ✨
- N dimensional array operations
  - Addition, multiplication, ReLU, matrix multiplication, ...
- Autograd system
- GPU acceleration with CUDA
- Deep neural networks
## Example: MNIST classifier 🔢
See ```"Showcase/MNIST/"```<br>
```cpp
class CNN : public Flow::Module
{

public:

    shared_ptr<Flow::Convolution> Conv1, Conv2;
    shared_ptr<Flow::Linear> Linear1, Linear2;

    CNN()
    {
        Conv1 = Flow::Convolution::Create( 1, 10, { 5, 5 } );
        Conv2 = Flow::Convolution::Create( 10, 20, { 5, 5 } );
        Linear1 = Flow::Linear::Create( { 320, 50 }, { 50 } );
        Linear2 = Flow::Linear::Create( { 50, 10 }, { 10 } );
        Modules = { Conv1, Conv2, Linear1, Linear2 };
    }

    NARRAY Forward( NARRAY arr ) override
    {
        NARRAY a1 = Flow::Unsqueeze( arr, 1 );
        NARRAY a2 = Flow::ReLU( MaxPool2d( Conv1->Forward(a1), { 2, 2 } ) );
        NARRAY a3 = Flow::ReLU( MaxPool2d( Conv2->Forward(a2), { 2, 2 } ) );
        NARRAY a4 = Flow::Reshape( a3, { a3->GetShape()[0], 320 } );
        NARRAY a5 = Flow::ReLU( Linear1->Forward(a4) );
        NARRAY a6 = Linear2->Forward(a5);
        return Flow::Softmax( a6, 1 );
    }

};

int main()
{
    // ...

    CNN network;
    Flow::Optimizer optimizer( network.GetParameters(), 0.001f, 1e-8f, 0.0f );

    for ( int epoch = 0; epoch < 10; epoch++ )
    {
        auto batches = Flow::CreateBatches( xTrain, yTrain, 100 );
        for ( auto batch : batches )
        {
            NARRAY yPredicted = network.Forward(batch.first);
            NARRAY loss = Flow::CrossEntropy( yPredicted, batch.second );
            optimizer.ZeroGrad();
            loss->Backpropagate();
            optimizer.Step();
        }
    }
}
```
<img src="chart1.png" />

## Performance 📈
```cpp
int batchSize = 100;
NARRAY arr1 = Flow::RandomUniform( { batchSize, 3, 4 }, -0.9999f, 0.9999f );
NARRAY arr2 = Flow::RandomUniform( { batchSize, 4, 5 }, -0.9999f, 0.9999f );
NARRAY result = Flow::Matmul( arr1, arr2 );
```
<img src="chart2.png" />
