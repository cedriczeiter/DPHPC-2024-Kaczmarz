cmake_minimum_required(VERSION 3.10)

project(Cuthill)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Define the source files
set(SOURCE_FILES
    cuthill.cpp
)

# Add the executable
add_executable(cuthill ${SOURCE_FILES})

# Link libraries
target_link_libraries(cuthill Eigen3::Eigen)