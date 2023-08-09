// Copyright (c) Juan M. G. de Agüero 2023

#include "Flow/Log.h"
#include "Flow/NArray.h"

#include "Showcase/SimpleNN.hpp"

#include "Test/TestAdd.hpp"
#include "Test/TestMult.hpp"

int main()
{
    Test_Add();
    Test_Mult();
    
    SimpleNN();
}