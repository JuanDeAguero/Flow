cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Flow VERSION 1.0)

add_executable(flow
    Main.cpp
    Source/NArray.cpp
    Source/NArrayCore.cpp
    Source/Operations/Add.cpp
    Source/Operations/Broadcast.cpp
    Source/Operations/Exp.cpp
    Source/Operations/Gather.cpp
    Source/Operations/Index.cpp
    Source/Operations/Log.cpp
    Source/Operations/Max.cpp
    Source/Operations/MM.cpp
    Source/Operations/Mul.cpp
    Source/Operations/Pow.cpp
    Source/Operations/ReLU.cpp
    Source/Operations/Reshape.cpp
    Source/Operations/Sum.cpp
    Source/Operations/Tanh.cpp
    Source/Operations/Transpose.cpp
    Source/Operations/Unsqueeze.cpp
    Source/Print.cpp
    Source/Vector.cpp
    Test/Torch.cpp
)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR}/Flow)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
target_link_libraries(flow PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flow PRIVATE ${TORCH_INCLUDE_DIRS})