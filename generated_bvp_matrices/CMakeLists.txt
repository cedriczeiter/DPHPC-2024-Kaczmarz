cmake_minimum_required(VERSION 3.10)
project(Cuthill LANGUAGES CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -fopenmp -g")

#################################
## Set variables
#################################

# Define source directory
set(SOURCE_DIR ${CMAKE_SOURCE_DIR})

# Define the source files
set(SOURCE_FILES
    ${SOURCE_DIR}/cuthill.cpp
)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Add the executable
add_executable(cuthill ${SOURCE_FILES})

# Link libraries
target_link_libraries(cuthill Eigen3::Eigen)

# Include directories
target_include_directories(cuthill PRIVATE ${EIGEN3_INCLUDE_DIR})
