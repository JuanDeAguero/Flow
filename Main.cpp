// Copyright (c) Juan M. G. de Agüero 2023

#include "Flow.h"

#include "Showcase/MakeMore.hpp"
#include "Showcase/SimpleNN.hpp"

#include "Test/TestAdd.hpp"
#include "Test/TestMult.hpp"

int main()
{
    Test_Add();
    Test_Mult();
    
    SimpleNN();

    //MakeMore();
}