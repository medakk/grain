cmake_minimum_required(VERSION 3.17)
project(grain LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find minifb
option(MINIFB_BUILD_EXAMPLES "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/minifb)

# Find fmt
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt)
find_package(fmt)

# Add source files. CMake takes care of using nvcc for .cu files
file(GLOB SRCFILES
        src/*.cpp
        src/*.cu)
add_executable(grain_bin ${SRCFILES})

# todo this seems specific to my GTX 1050. Need to figure out how to make it work anywhere
set_target_properties(grain_bin PROPERTIES CUDA_ARCHITECTURES "61")

target_include_directories(grain_bin PRIVATE
        "src/"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(grain_bin PRIVATE
        minifb
        fmt::fmt
        CUDA::cudart
        CUDA::cuda_driver)
