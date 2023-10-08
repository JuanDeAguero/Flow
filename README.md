# Flow
Machine Learning Library in C++
## Features
- N dimentional array operations
  - Addition, subtraction, multiplication, matrix multiplication, ...
- Autograd system
- Deep neural networks (WIP)
- GPU acceleration (WIP)
## MNIST classifier
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
Print one the train NArrays and its label.
```bash
for ( int i = 0; i < 28; i++ )
{
    for ( int j = 0; j < 28; j++ )
        cout << setw(3) << right << xTrain.Get({ 76, i * 28 + j }) << " ";
    cout << endl;
}
cout << "Label: " << trainLabels[76] << endl;
```
```bash
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0 
  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0 
  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0 
  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
Label: 5
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