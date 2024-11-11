cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX CUDA)

# Set C standard
set(CMAKE_C_STANDARD 11)
# Setup C++ standard
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -fopenmp")

############################################
#### Eigen install and include directory ###
############################################

include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0  # Specify a stable version
    )
FetchContent_MakeAvailable(eigen)
set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR}) # Set include directory for Eigen


############################################
#### Add Google Test for testing ###########
############################################

include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main 
    )
FetchContent_MakeAvailable(googletest)
# Set GTest and GMock libraries explicitly for linking
set(GTEST_BOTH_LIBRARIES gtest gmock)

############################################
#### Add and link the linear_systems library ###
############################################

# Add the linear_systems library and the header files for it
add_subdirectory(${CMAKE_SOURCE_DIR}/../linear_systems ${CMAKE_BINARY_DIR}/linear_systems_build)

# Set up the include directory for linear_systems explicitly
set(LINEAR_SYSTEMS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../linear_systems/src/linear_systems)

############################################
#### Which sources to build             ####
############################################

# Set up the source directory and the sources
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SOURCES
  ${SOURCE_DIR}/solvers/banded.cpp
  ${SOURCE_DIR}/solvers/basic.cpp
  ${SOURCE_DIR}/solvers/common.cpp
  ${SOURCE_DIR}/solvers/random.cpp
  ${SOURCE_DIR}/solvers/cuda_basic.cpp
  ${SOURCE_DIR}/solvers/cuda_funcs.cu
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

# Create an object library for the shared C++ sources
add_library(shared_sources OBJECT ${CXX_SOURCES})

# Set include directories for shared sources (like .hpp files)
target_include_directories(shared_sources PRIVATE ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/../linear_systems/src/linear_systems ${SOURCE_DIR}/solvers)


############################################
#### Add executables and link libraries ####
############################################

# Set include directories for all executables
set(EXECUTABLE_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${LINEAR_SYSTEMS_INCLUDE_DIR} ${SOURCE_DIR}/solvers)


# Create executables for your different benchmarks
add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking_convergence.cpp)
add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp)
add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp)
add_executable(testing ${SOURCE_DIR}/testing.cpp)
add_executable(benchmark_one ${SOURCE_DIR}/benchmark_one.cpp)

# Link C++ executables with linear_systems, shared_sources, and GoogleTest for testing
target_include_directories(benchmark_convergence PRIVATE ${LINEAR_SYSTEMS_INCLUDE_DIR} ${EXECUTABLE_INCLUDE_DIRS})
target_include_directories(benchmark_executable PRIVATE ${LINEAR_SYSTEMS_INCLUDE_DIR} ${EXECUTABLE_INCLUDE_DIRS})
target_include_directories(kaczmarz_example PRIVATE ${LINEAR_SYSTEMS_INCLUDE_DIR} ${EXECUTABLE_INCLUDE_DIRS})
target_include_directories(testing PRIVATE ${LINEAR_SYSTEMS_INCLUDE_DIR} ${EXECUTABLE_INCLUDE_DIRS})
target_include_directories(benchmark_one PRIVATE ${LINEAR_SYSTEMS_INCLUDE_DIR} ${EXECUTABLE_INCLUDE_DIRS})

target_link_libraries(benchmark_convergence PRIVATE linear_systems $<TARGET_OBJECTS:shared_sources>)
target_link_libraries(benchmark_executable PRIVATE linear_systems $<TARGET_OBJECTS:shared_sources>)
target_link_libraries(kaczmarz_example PRIVATE linear_systems $<TARGET_OBJECTS:shared_sources>)
target_link_libraries(testing PRIVATE linear_systems gtest $<TARGET_OBJECTS:shared_sources>)
target_link_libraries(benchmark_one PRIVATE linear_systems $<TARGET_OBJECTS:shared_sources>)


############################################
#### CUDA support and executables ##########
############################################

# Enable CUDA support
find_package(CUDA 11.7 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86)

# Add CUDA executables
include_directories(${CMAKE_SOURCE_DIR}/src/solvers)
add_executable(testing_cuda ${CUDA_SOURCES} src/main_cuda.cpp)

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
