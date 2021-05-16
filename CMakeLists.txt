cmake_minimum_required(VERSION 3.17)
project(grain LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set (CMAKE_CXX_STANDARD 17)

file(GLOB SRCFILES src/*.cpp)
add_executable(grain_bin ${SRCFILES})

target_include_directories(grain_bin PRIVATE
        "src/"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(grain_bin PRIVATE
        CUDA::cudart
        CUDA::cuda_driver)
