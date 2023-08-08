cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Flow VERSION 1.0)

add_executable(flow
    Main.cpp
    Source/Log.cpp
    Source/NArray.cpp
)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Include)

target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR}/Flow)