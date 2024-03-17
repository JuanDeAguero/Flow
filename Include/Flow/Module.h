// Copyright (c) 2023 Juan M. G. de Agüero

#pragma once

#include <memory>
#include <vector>

#include "Flow/NArray.h"

namespace Flow
{
    using namespace std;

    class Module
    {

    public:

        Module();

        virtual NARRAY Forward( NARRAY arr );

        virtual vector<NARRAY> GetParameters();

    protected:

        vector<shared_ptr<Module>> Modules;

    };

    class Linear : public Module
    {

    public:

        Linear();

        Linear( vector<int> weightShape, vector<int> biasShape );

        static shared_ptr<Linear> Create( vector<int> weightShape, vector<int> biasShape );

        NARRAY Forward( NARRAY arr ) override;

        vector<NARRAY> GetParameters() override;

    private:

        NARRAY Weight;

        NARRAY Bias;

    };

    class Conv2d : public Module
    {

    };
}