cmake_minimum_required(VERSION 3.10)
project(linear_systems)

set(CMAKE_CXX STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(linear_systems STATIC src/linear_systems/dense.cpp src/linear_systems/sparse.cpp)

target_link_libraries(linear_systems Eigen3::Eigen)

target_include_directories(linear_systems PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
