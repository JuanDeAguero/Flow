## The Concept of Stride
Stride $s$ for an $N$-dimensional tensor helps to map its multi-dimensional index to a single dimensional index in memory. For an $N$-dimensional tensor with shape $\left(d_1, d_2, \ldots, d_N\right)$, stride is defined for each dimension.

## Computation of Stride
The stride of a tensor is computed based on its shape. We calculate the stride in reverse order, from the last dimension to the first. For the last dimension (the innermost), the stride is always 1. As we move to outer dimensions, we multiply the current stride by the size of the current dimension.

Given a tensor shape $\left(d_1, d_2, \ldots, d_N\right)$, the stride $s$ for each dimension is calculated as:
```math
$$
\begin{aligned}
s_N & =1 \\
s_{N-1} & =s_N \times d_N \\
s_{N-2} & =s_{N-1} \times d_{N-1} \\
\vdots & \\
s_1 & =s_2 \times d_2
\end{aligned}
$$
```

## Example
Consider a 2D tensor (matrix) with shape $(3,4)$ :
```math
$$
\left[\begin{array}{llll}
a & b & c & d \\
e & f & g & h \\
i & j & k & l
\end{array}\right]
$$
```
The stride for the second dimension (columns) is $s_2=1$ since you move one step in memory from $a$ to $b$. For the first dimension (rows), $s_1=s_2 \times d_2=1 \times 4=4$ as you jump 4 steps in memory to move from $a$ to $e$.

## Indexing using Stride
To find the memory location (1D index) of an element in the tensor given its multidimensional coordinates $\left(c_1, c_2, \ldots, c_N\right)$, we use:
```math
$$
\text { index }=\sum_{i=1}^N c_i \times s_i
$$
```
This equation gives the offset in the single-dimensional memory layout based on the multidimensional coordinates of the element.
