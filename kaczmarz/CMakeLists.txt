cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX CUDA)

# Set C standard
set(CMAKE_C_STANDARD 11)
# Setup C++ standard
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")

############################################
#### Eigen install and include directory ###
############################################

# Fetch Eigen if it's not installed
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0  # You can specify a stable version here
)
FetchContent_MakeAvailable(eigen)

# Add Eigen include directory
include_directories(${eigen_SOURCE_DIR})

############################################
#### Which sources to build and include ####
############################################

# Set up the source directory and the sources
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SOURCES
  ${SOURCE_DIR}/solvers/banded.cpp
  ${SOURCE_DIR}/solvers/basic.cpp
  ${SOURCE_DIR}/solvers/common.cpp
  ${SOURCE_DIR}/solvers/random.cpp
  ${SOURCE_DIR}/solvers/basic_cuda.cu
)

# Separate .cu files and .cpp files for different compilers
set(CXX_SOURCES)
set(CUDA_SOURCES)

# Split the sources into C++ and CUDA files
foreach(src ${SOURCES})
    if(src MATCHES ".cu$")
        list(APPEND CUDA_SOURCES ${src})
    else()
        list(APPEND CXX_SOURCES ${src})
    endif()
endforeach()

############################################
#### Add Google Test for testing ###########
############################################

# Fetch GoogleTest from the official GitHub repository
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main  # Use the latest stable version tag, or you can use "main" for the latest commit
)
FetchContent_MakeAvailable(googletest)

############################################
#### Add and link the linear_systems library ###
############################################

# Add the linear_systems library from a relative path
add_subdirectory(${CMAKE_SOURCE_DIR}/../linear_systems ${CMAKE_BINARY_DIR}/linear_systems_build)
include_directories(${CMAKE_SOURCE_DIR}/../linear_systems/src/linear_systems)

############################################
#### Add executables and link libraries ####
############################################

# Create executables for your different benchmarks
add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking_convergence.cpp ${CXX_SOURCES})
add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp ${CXX_SOURCES})
add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp ${CXX_SOURCES})
add_executable(testing ${SOURCE_DIR}/testing.cpp ${CXX_SOURCES})
add_executable(benchmark_one ${SOURCE_DIR}/benchmark_one.cpp ${CXX_SOURCES})

# Link C++ executables with linear_systems and GoogleTest for testing
target_link_libraries(benchmark_convergence PRIVATE linear_systems)
target_link_libraries(benchmark_executable PRIVATE linear_systems)
target_link_libraries(kaczmarz_example PRIVATE linear_systems)
target_link_libraries(testing PRIVATE linear_systems gtest)
target_link_libraries(benchmark_one PRIVATE linear_systems)

############################################
#### CUDA support and executables ##########
############################################

# Enable CUDA support
find_package(CUDA 11.7 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86)

# Add CUDA executables
add_executable(testing_cuda ${CUDA_SOURCES} src/main_cuda.cpp)

# Ensure the linear_systems library is built
add_library(linear_systems STATIC ${SOURCE_DIR}/linear_systems/some_source_file.cpp)

# Link the CUDA executables with the necessary libraries
target_link_libraries(testing_cuda PRIVATE linear_systems)

############################################
#### Configure and copy Python plot files ##
############################################

# Configure and copy Python plot files
configure_file(${SOURCE_DIR}/plot.py ${CMAKE_BINARY_DIR}/plot.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_convergence.py ${CMAKE_BINARY_DIR}/plot_convergence.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_dense.py ${CMAKE_BINARY_DIR}/plot_dense.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_sparse.py ${CMAKE_BINARY_DIR}/plot_sparse.py COPYONLY)
