cmake_minimum_required(VERSION 3.10)
project(kaczmarz)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -fopenmp -g")

#################################
## Download dependencies
#################################

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

#################################
## Set variables
#################################

# Define source directory
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(NON_CUDA_SOURCES
  ${SOURCE_DIR}/solvers/basic.cpp
  ${SOURCE_DIR}/solvers/common.cpp
  ${SOURCE_DIR}/solvers/random.cpp
)

# All files that have something that needs cuda inside in here
set(CUDA_SOURCES
  ${SOURCE_DIR}/solvers/banded.cu
  ${SOURCE_DIR}/solvers/banded.cpp
)

################################
## Include the right folders (searches there for headers)
################################

include_directories(
  ${CMAKE_SOURCE_DIR}/../linear_systems
  ${CMAKE_SOURCE_DIR}/../linear_systems/src
  ${SOURCE_DIR}/solvers
)

# Add subdirectory for linear_systems
add_subdirectory(../linear_systems linear_systems_build)

################################
## Build test and general library which is always run
################################

# Create non-CUDA library
add_library(non_cuda_library STATIC ${NON_CUDA_SOURCES})
target_link_libraries(non_cuda_library PRIVATE Eigen3::Eigen)

# Always build the `testing` target (only with non-CUDA library)
add_executable(testing ${SOURCE_DIR}/testing.cpp)
target_link_libraries(testing PRIVATE non_cuda_library linear_systems gtest)

################################
## Now stuff that needs Cuda follows
################################

# Conditionally add CUDA library and executables if CUDA is available
if (CMAKE_CUDA_COMPILER)
  # write out the CUDA compiler
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  # Set CUDA architectures (to avoid CMP0104 warnings)
  if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
  endif()
  # Allow many relevant CUDA architectures
  set(CUDA_ARCHITECTURES 60 61 70 75 80 86)
  
  # Create CUDA library
  add_library(cuda_library STATIC ${CUDA_SOURCES})
  set_target_properties(cuda_library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(cuda_library PRIVATE Eigen3::Eigen)

  # CUDA-enabled executables
  add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking_convergence.cpp)
  add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp)
  add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp)
  add_executable(benchmark_one ${SOURCE_DIR}/benchmark_one.cpp)

  # Link the non-CUDA and CUDA libraries to the executables
  target_link_libraries(benchmark_convergence PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_executable PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(kaczmarz_example PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_one PRIVATE non_cuda_library cuda_library linear_systems)

else()
  message(STATUS "No CUDA compiler found, skipping CUDA targets")
endif()

################################
## Copy files to build directory for Python stuff
################################

# Configure files
CONFIGURE_FILE(${SOURCE_DIR}/plot.py ${PATH_TO_BUILD} plot.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_convergence.py ${PATH_TO_BUILD} plot_convergence.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_dense.py ${PATH_TO_BUILD} plot_dense.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_sparse.py ${PATH_TO_BUILD} plot_sparse.py COPYONLY)


