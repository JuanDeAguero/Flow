# Flow
Machine Learning Library in C++

Features:
- N dimentional array operations
  - Addition, subtraction, multiplication, matrix multiplication, ...
- Autograd system
- Deep neural networks
- GPU acceleration

Example:
```bash
Flow::NArray arr1 = Flow::Create( { 3, 3 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8 } );
Flow::NArray arr2 = Flow::Create( { 3 }, { 1, 10, 100 } );
Flow::NArray arr3 = Flow::Add( arr1, arr2 );
```