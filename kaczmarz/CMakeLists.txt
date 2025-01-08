cmake_minimum_required(VERSION 3.10)
project(kaczmarz LANGUAGES CXX)

# Enable CUDA only if it is available on the system
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

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
  ${SOURCE_DIR}/solvers/sparse_cg.cpp
)

# All files that have something that needs cuda inside in here
set(CUDA_SOURCES
  ${SOURCE_DIR}/solvers/cuda_common.cu
  ${SOURCE_DIR}/solvers/cuda_utils.cpp
  ${SOURCE_DIR}/solvers/banded.cu
  ${SOURCE_DIR}/solvers/basic.cpp
  ${SOURCE_DIR}/solvers/banded.cpp
  ${SOURCE_DIR}/solvers/nolpde.cpp
  ${SOURCE_DIR}/solvers/nolpde.cu
  ${SOURCE_DIR}/solvers/asynchronous.cu
  ${SOURCE_DIR}/solvers/asynchronous.cpp
  ${SOURCE_DIR}/solvers/carp.cu
  ${SOURCE_DIR}/solvers/carp.cpp
  ${SOURCE_DIR}/solvers/carp_utils.cu
  ${SOURCE_DIR}/solvers/cusolver.cpp
)

################################
## Include the right folders (searches there for headers)
################################

include_directories(
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
#add_executable(testing ${SOURCE_DIR}/testing.cpp)
#target_link_libraries(testing PRIVATE non_cuda_library linear_systems gtest)

################################
## Now stuff that needs Cuda follows
################################

# Conditionally add CUDA library and executables if CUDA is available
if (CMAKE_CUDA_COMPILER)
  # write out the CUDA compiler
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

  # Setting to use dynamic linking of the CUDA runtime library
  # -> avoids "nvlink warning : Skipping incompatible something_something.a ..." warnings
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

  # Set CUDA architectures (to avoid CMP0104 warnings)
  set(CMAKE_CUDA_ARCHITECTURES 86)

  include_directories(${CUDA_INCLUDE_DIRS})
  find_package(cudss REQUIRED)
  
  # Create CUDA library
  add_library(cuda_library STATIC ${CUDA_SOURCES})
  set_target_properties(cuda_library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(cuda_library PRIVATE Eigen3::Eigen cusparse cusolver cudss)

  # CUDA-enabled executables
  add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking/benchmark_convergence.cpp)
  add_executable(benchmark_single_run ${SOURCE_DIR}/benchmarking/benchmark_single_run.cpp)
  add_executable(benchmark_nolpde ${SOURCE_DIR}/benchmarking/benchmark_nolpde.cpp)
  add_executable(benchmark_banded ${SOURCE_DIR}/benchmarking/benchmark_banded.cpp)
  add_executable(benchmark_carp_multiple_run ${SOURCE_DIR}/benchmarking/benchmark_carp_multiple_run.cpp)
  add_executable(relaxation_parameter_carp_single_run ${SOURCE_DIR}/benchmarking/relaxation_parameter_carp_single_run.cpp)
  add_executable(relaxation_parameter_carp_multiple_run ${SOURCE_DIR}/benchmarking/relaxation_parameter_carp_multiple_run.cpp)
  add_executable(benchmark_pde ${SOURCE_DIR}/benchmarking/benchmark_pde.cpp)
  add_executable(show_convergence ${SOURCE_DIR}/benchmarking/show_convergence.cpp)

  # Link the non-CUDA and CUDA libraries to the executables
  target_link_libraries(benchmark_convergence PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_single_run PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_nolpde PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_banded PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_carp_multiple_run PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(relaxation_parameter_carp_single_run PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(benchmark_pde PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(show_convergence PRIVATE non_cuda_library cuda_library linear_systems)
  target_link_libraries(relaxation_parameter_carp_multiple_run PRIVATE non_cuda_library cuda_library linear_systems)


else()
  message(STATUS "No CUDA compiler found, skipping CUDA targets")
endif()

################################
## Copy files to build directory for Python stuff
################################

# Configure files
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot.py ${PATH_TO_BUILD} plot.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot_convergence.py ${PATH_TO_BUILD} plot_convergence.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot_dense.py ${PATH_TO_BUILD} plot_dense.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot_sparse.py ${PATH_TO_BUILD} plot_sparse.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot_sparse_pde.py ${PATH_TO_BUILD} plot_sparse_pde.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plotting/plot_convergence_new.py ${PATH_TO_BUILD} plot_convergence_new.py COPYONLY)
