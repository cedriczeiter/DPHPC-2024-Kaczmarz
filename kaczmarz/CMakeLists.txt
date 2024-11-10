cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX CUDA)

# Set C standard
set(CMAKE_C_STANDARD 11)
# Setup C++ standard
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -fopenmp")

# Allow users to specify the Eigen installation path
option(EIGEN_PATH "Path to Eigen library" "${CMAKE_SOURCE_DIR}/../eigen3")
include_directories(${EIGEN_PATH})

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

# Fetch GoogleTest from the official GitHub repository
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main # Use the latest stable version tag, or you can use "main" for the latest commit
)

# This will make sure the content is downloaded and added
FetchContent_MakeAvailable(googletest)

# Create executables for your different benchmarks
add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking_convergence.cpp ${CXX_SOURCES})
add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp ${CXX_SOURCES})
add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp ${CXX_SOURCES})
add_executable(testing ${SOURCE_DIR}/testing.cpp ${CXX_SOURCES})
add_executable(benchmark_one ${SOURCE_DIR}/benchmark_one.cpp ${CXX_SOURCES})

# Add necessary include directories for linear_systems
target_include_directories(kaczmarz_example PRIVATE ${SOURCE_DIR}/linear_systems)
target_include_directories(testing PRIVATE ${SOURCE_DIR}/linear_systems)
target_include_directories(benchmark_executable PRIVATE ${SOURCE_DIR}/linear_systems)
target_include_directories(benchmark_one PRIVATE ${SOURCE_DIR}/linear_systems)
target_include_directories(benchmark_convergence PRIVATE ${SOURCE_DIR}/linear_systems)

# Link C++ executables with linear_systems
target_link_libraries(kaczmarz_example PRIVATE linear_systems)
target_link_libraries(testing PRIVATE linear_systems gtest)
target_link_libraries(benchmark_executable PRIVATE linear_systems)
target_link_libraries(benchmark_convergence PRIVATE linear_systems)
target_link_libraries(benchmark_one PRIVATE linear_systems)

# Enable CUDA support
find_package(CUDA 11.7 REQUIRED)

# Add CUDA executables without cuda_add_executable
add_executable(kaczmarz_example_cuda ${CUDA_SOURCES})
add_executable(testing_cuda ${CUDA_SOURCES})
add_executable(benchmark_executable_cuda ${CUDA_SOURCES})
add_executable(benchmark_convergence_cuda ${CUDA_SOURCES})
add_executable(benchmark_one_cuda ${CUDA_SOURCES})

# Link the CUDA executables with the necessary libraries using plain syntax
target_link_libraries(kaczmarz_example_cuda linear_systems)
target_link_libraries(testing_cuda linear_systems gtest)
target_link_libraries(benchmark_executable_cuda linear_systems)
target_link_libraries(benchmark_convergence_cuda linear_systems)
target_link_libraries(benchmark_one_cuda linear_systems)

# Configure and copy Python plot files
configure_file(${SOURCE_DIR}/plot.py ${CMAKE_BINARY_DIR}/plot.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_convergence.py ${CMAKE_BINARY_DIR}/plot_convergence.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_dense.py ${CMAKE_BINARY_DIR}/plot_dense.py COPYONLY)
configure_file(${SOURCE_DIR}/plot_sparse.py ${CMAKE_BINARY_DIR}/plot_sparse.py COPYONLY)

# Add the linear_systems subdirectory
add_subdirectory(../linear_systems linear_systems_build)
