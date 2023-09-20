// Copyright (c) Juan M. G. de Agüero 2023

#include "Flow.h"

#include "Showcase/MakeMore/MakeMore.hpp"
#include "Showcase/SimpleNN.hpp"
#include "Showcase/MNIST/MNIST.hpp"

#include "Test/TestAdd.hpp"
#include "Test/TestBackward.hpp"
#include "Test/TestMul.hpp"
#include "Test/TestSum.hpp"

int main()
{
    Test_Backward();
    Test_Add();
    Test_Mul();
    //Test_Sum();
    //SimpleNN();
    //MakeMore();
    //MNIST();
}