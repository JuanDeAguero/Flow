cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Flow VERSION 1.0)

add_executable(flow
    Main.cpp
    Source/Log.cpp
    Source/NArray.cpp
    Source/NArrayCore.cpp
)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_include_directories(flow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR})
target_include_directories(flow PRIVATE ${INCLUDE_DIR}/Flow)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
target_link_libraries(flow PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flow PRIVATE ${TORCH_INCLUDE_DIRS})