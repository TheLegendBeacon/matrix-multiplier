cmake_minimum_required(VERSION 3.15)
project(matrix_multiplier VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

add_library(matrix_engine
    src/matrix.cpp
include/matrix.hpp
)

target_include_directories(matrix_engine PUBLIC include)

add_executable(matrix_app apps/main.cpp)

target_link_libraries(matrix_app PRIVATE matrix_engine)