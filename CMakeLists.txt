cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(Flow VERSION 1.0 LANGUAGES CXX CUDA)
find_package(CUDA 11.8 REQUIRED)
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cu)
add_library(flow STATIC ${SOURCES})
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include/Flow)
target_link_libraries(flow PRIVATE ${CUDA_LIBRARIES})
target_include_directories(flow PRIVATE ${CUDA_INCLUDE_DIRS})