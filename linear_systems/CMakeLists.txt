cmake_minimum_required(VERSION 3.10)
project(linear_systems)

set(CMAKE_CXX STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Download Eigen
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
set(EIGEN_BUILD_TESTS OFF CACHE BOOL "Disable Eigen tests to suppress output" FORCE)
FetchContent_MakeAvailable(eigen)
# Print that Eigen now installed
message(STATUS "Eigen installed")

# Download Gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(googletest)
# Print that Gtest now installed
message(STATUS "Gtest installed")

add_library(linear_systems STATIC src/linear_systems/dense.cpp src/linear_systems/sparse.cpp)

target_link_libraries(linear_systems Eigen3::Eigen)

target_include_directories(linear_systems PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
