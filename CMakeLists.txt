cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Flow VERSION 1.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_library(flow STATIC
    Source/CUDA/Add.cu
    Source/CUDA/Broadcast.cu
    Source/CUDA/Exp.cu
    Source/CUDA/Gather.cu
    Source/CUDA/Index.cu
    Source/CUDA/Log.cu
    Source/CUDA/Max.cu
    Source/CUDA/MM.cu
    Source/CUDA/Mul.cu
    Source/CUDA/Pow.cu
    Source/CUDA/ReLU.cu
    Source/CUDA/Sum.cu
    Source/CUDA/Tanh.cu
    Source/NArray.cpp
    Source/NArrayCore.cpp
    Source/Operations/Add.cpp
    Source/Operations/BMM.cpp
    Source/Operations/Broadcast.cpp
    Source/Operations/Dot.cpp
    Source/Operations/Exp.cpp
    Source/Operations/Gather.cpp
    Source/Operations/Index.cpp
    Source/Operations/Log.cpp
    Source/Operations/Matmul.cpp
    Source/Operations/Max.cpp
    Source/Operations/MM.cpp
    Source/Operations/Mul.cpp
    Source/Operations/MV.cpp
    Source/Operations/Pow.cpp
    Source/Operations/Prod.cpp
    Source/Operations/ReLU.cpp
    Source/Operations/Reshape.cpp
    Source/Operations/Squeeze.cpp
    Source/Operations/Sum.cpp
    Source/Operations/Tanh.cpp
    Source/Operations/Transpose.cpp
    Source/Operations/Unsqueeze.cpp
    Source/Optimizer.cpp
    Source/Print.cpp
    Source/Vector.cpp
)

target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include/Flow)

target_link_libraries(flow PRIVATE ${CUDA_LIBRARIES})
target_include_directories(flow PRIVATE ${CUDA_INCLUDE_DIRS})