cmake_minimum_required(VERSION 3.18)

project(Flow VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(FLOW_CUDA_ROOT "" CACHE PATH "Path to the CUDA Toolkit (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1)")
option(FLOW_CUDA_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc (needed for newer MSVC toolsets)" ON)

if(FLOW_CUDA_ROOT)
    if(NOT EXISTS "${FLOW_CUDA_ROOT}/bin/nvcc.exe")
        message(FATAL_ERROR "FLOW_CUDA_ROOT was set to '${FLOW_CUDA_ROOT}', but nvcc.exe was not found at '${FLOW_CUDA_ROOT}/bin/nvcc.exe'.\n")
    endif()
    set(CUDAToolkit_ROOT "${FLOW_CUDA_ROOT}" CACHE PATH "" FORCE)
    if(EXISTS "${FLOW_CUDA_ROOT}/bin/nvcc.exe" AND NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${FLOW_CUDA_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
    endif()
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS "Visual Studio generator detected: ${CMAKE_GENERATOR}")
endif()

if(FLOW_CUDA_ALLOW_UNSUPPORTED_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

file(GLOB_RECURSE FLOW_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cu)

add_library(flow STATIC ${FLOW_SOURCES})

target_include_directories(flow
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/Include
        ${CMAKE_CURRENT_SOURCE_DIR}/Include/Flow
)

target_link_libraries(flow
    PUBLIC
        CUDA::cudart
)

option(FLOW_BUILD_TESTS "Build Flow tests" ON)
option(FLOW_BUILD_MNIST "Build MNIST showcase" ON)

if(FLOW_BUILD_TESTS)
    add_subdirectory(Test)
endif()

if(FLOW_BUILD_MNIST)
    add_subdirectory(Showcase/MNIST)
endif()