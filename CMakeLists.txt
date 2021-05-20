cmake_minimum_required(VERSION 3.17)
project(grain LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find minifb
option(MINIFB_BUILD_EXAMPLES "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/minifb)

# Find fmt
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt)
find_package(fmt)

# Find google benchmark
if(GRAIN_BENCHMARK)
    option(BENCHMARK_ENABLE_GTEST_TESTS "" OFF)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/benchmark)
endif()

# List of dependencies to link to that are common for all targets
set(GRAIN_DEPS minifb fmt::fmt CUDA::cudart CUDA::cuda_driver)

###########################
# Build library grain_lib
###########################

# Add source files. CMake takes care of using nvcc for .cu files
file(GLOB LIBFILES
        lib/*.cpp
        lib/*.cu)
add_library(grain_lib ${LIBFILES})

# todo this seems specific to my GTX 1050. Need to figure out how to make it work anywhere
set_target_properties(grain_lib PROPERTIES CUDA_ARCHITECTURES "61")

target_include_directories(grain_lib PRIVATE
        "lib/"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(grain_lib PRIVATE ${GRAIN_DEPS})

##############################
# Build executable grain_bin
##############################

add_executable(grain_bin src/main.cpp)
target_include_directories(grain_bin PRIVATE
        "lib/" # TODO better way to expose headers.
        "src/")
target_link_libraries(grain_bin PRIVATE
        grain_lib
        ${GRAIN_DEPS})

###############################
# Build benchmarks grain_bench
###############################
if(GRAIN_BENCHMARK)
    add_executable(grain_bench bench/bench.cpp)
    target_include_directories(grain_bench PRIVATE
            "lib/" # TODO better way to expose headers.
            "bench/")
    target_link_libraries(grain_bench PRIVATE
            grain_lib
            ${GRAIN_DEPS}
            benchmark::benchmark)
endif()