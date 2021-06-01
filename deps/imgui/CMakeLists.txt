cmake_minimum_required(VERSION 3.1)
project(imgui)

find_package(OpenGL REQUIRED)

file(GLOB IMGUI_SRCFILES
        *.cpp
        backends/*.cpp)
add_library(imgui ${IMGUI_SRCFILES})
target_include_directories(imgui PUBLIC "." "backends/" "../glfw/include/")
target_link_libraries(imgui ${OPENGL_LIBRARIES})
