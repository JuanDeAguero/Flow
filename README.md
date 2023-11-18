# Flow
Machine Learning Library in C++
## Features
- N dimensional array operations
  - Addition, subtraction, multiplication, matrix multiplication, ...
- Autograd system
- GPU acceleration with CUDA
- Deep neural networks
## Example: MNIST classifier
Load the data.
```bash
vector<float> trainImages = ReadImagesMNIST("...");
vector<float> trainLabels = ReadLabelsMNIST("...");
vector<float> testImages = ReadImagesMNIST("...");
vector<float> testLabels = ReadLabelsMNIST("...");

trainImages.resize( 784 * 100 );
trainLabels.resize( 100 );
testImages.resize( 784 * 100 );
testLabels.resize( 100 );
```
Create the train and test NArrays.
```bash
Flow::NArray xTrain = Flow::Create( { 100, 784 }, trainImages );
Flow::NArray yTrain = Flow::Create( { 100 }, trainLabels );
Flow::NArray xTest = Flow::Create( { 100, 784 }, testImages );
Flow::NArray yTest = Flow::Create( { 100 }, trainLabels );
```
Create random weights and biases.
```bash
Flow::NArray w1 = Flow::Random({ 784, 128 });
Flow::NArray b1 = Flow::Random({ 128 });
Flow::NArray w2 = Flow::Random({ 128, 10 });
Flow::NArray b2 = Flow::Random({ 10 });
```
Set a learing rate of 0.1.
```bash
float learningRate = 0.1f;
```
100 iterations.
```bash
for ( int epoch = 0; epoch < 100; epoch++ )
{
```
Simple two-layer network. 784 -> 128 -> 10<br>
ReLU is the activation function.<br>
Cross entropy is the loss function.
```math
\begin{align*}
a &= \text{ReLU}(\text{xTrain} \cdot w1 + b1) \\
\text{yPred} &= a \cdot w2 + b2 \\
\text{loss} &= - \sum_{i} \text{yTrain}_i \log(\text{yPred}_i)
\end{align*}
```
```bash
    Flow::NArray a = ReLU( Add( MM( xTrain, w1 ), b1 ) );
    Flow::NArray yPred = Add( MM( a, w2 ), b2 );
    Flow::NArray loss = CrossEntropy( yPred, yTrain );
```
Reset the gradients and backpropagate the loss.<br>
The autograd will do its magic.
```bash
    w1.GetGradient().Reset(0);
    b1.GetGradient().Reset(0);
    w2.GetGradient().Reset(0);
    b2.GetGradient().Reset(0);

    loss.Backpropagate();
```
Gradient descent.<br>
Modify the weights and biases using their gradients and the learning rate.
```bash
    w1 = Sub( w1.Copy(), Mul( w1.GetGradient().Copy(), learningRate ) );
    b1 = Sub( b1.Copy(), Mul( b1.GetGradient().Copy(), learningRate ) );
    w2 = Sub( w2.Copy(), Mul( w2.GetGradient().Copy(), learningRate ) );
    b2 = Sub( b2.Copy(), Mul( b2.GetGradient().Copy(), learningRate ) );
```
Print the loss in every epoch.
```bash
    Flow::Print(loss);
}
```
Equivalent code using pytorch:<br>
https://colab.research.google.com/drive/1GwRjaX5Jh4rTxrPH9ChfaPl-YTaznoIn?usp=sharing