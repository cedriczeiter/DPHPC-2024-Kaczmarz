cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX CUDA)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread -fopenmp -g")

# download eigen
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)


# Define source directories
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(NON_CUDA_SOURCES
  ${SOURCE_DIR}/solvers/basic.cpp
  ${SOURCE_DIR}/solvers/common.cpp
  ${SOURCE_DIR}/solvers/random.cpp
)

include_directories(
  ${CMAKE_SOURCE_DIR}/linear_systems
  ${CMAKE_SOURCE_DIR}/linear_systems/src
  ${SOURCE_DIR}/solvers
)
# Fetch GoogleTest from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(googletest)

# Always build the `testing` target (only with non-CUDA sources)
add_executable(testing ${SOURCE_DIR}/testing.cpp ${NON_CUDA_SOURCES})
target_link_libraries(testing PRIVATE Eigen3::Eigen linear_systems gtest) # this part is a weird dependency, because we need to build the linear_systems cmake first

# Conditionally add CUDA executables
if (CMAKE_CUDA_COMPILER)
 #allow all relevant cuda architectures
  set(CUDA_ARCHITECTURES 60 61 70 75 80 86)

  set(CUDA_SOURCES
    ${SOURCE_DIR}/solvers/banded.cu
    ${SOURCE_DIR}/solvers/banded.cpp
  )
  
  
  # CUDA-enabled executables
  add_executable(benchmark_convergence ${SOURCE_DIR}/benchmarking_convergence.cpp ${NON_CUDA_SOURCES} ${CUDA_SOURCES})
  add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp ${NON_CUDA_SOURCES} ${CUDA_SOURCES})
  add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp ${NON_CUDA_SOURCES} ${CUDA_SOURCES} )
  add_executable(benchmark_one ${SOURCE_DIR}/benchmark_one.cpp ${NON_CUDA_SOURCES} ${CUDA_SOURCES})

  # Set CUDA-specific properties
  set_target_properties(kaczmarz_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(benchmark_executable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(benchmark_convergence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(benchmark_one PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Include directories for CUDA targets
  target_include_directories(kaczmarz_example PRIVATE ${CMAKE_SOURCE_DIR}/linear_systems ${CMAKE_SOURCE_DIR}/linear_systems/src ${SOURCE_DIR}/solvers) 
  target_include_directories(benchmark_executable PRIVATE ${CMAKE_SOURCE_DIR}/linear_systems ${CMAKE_SOURCE_DIR}/linear_systems/src ${SOURCE_DIR}/solvers)
  target_include_directories(benchmark_convergence PRIVATE ${CMAKE_SOURCE_DIR}/linear_systems ${CMAKE_SOURCE_DIR}/linear_systems/src ${SOURCE_DIR}/solvers)
  target_include_directories(benchmark_one PRIVATE ${CMAKE_SOURCE_DIR}/linear_systems ${CMAKE_SOURCE_DIR}/linear_systems/src ${SOURCE_DIR}/solvers)

  # Link libraries for CUDA targets
  target_link_libraries(kaczmarz_example PRIVATE Eigen3::Eigen linear_systems )
  target_link_libraries(benchmark_executable PRIVATE Eigen3::Eigen linear_systems )
  target_link_libraries(benchmark_convergence PRIVATE Eigen3::Eigen linear_systems )
  target_link_libraries(benchmark_one PRIVATE Eigen3::Eigen linear_systems )
endif()

# Configure files
CONFIGURE_FILE(${SOURCE_DIR}/plot.py ${PATH_TO_BUILD} plot.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_convergence.py ${PATH_TO_BUILD} plot_convergence.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_dense.py ${PATH_TO_BUILD} plot_dense.py COPYONLY)
CONFIGURE_FILE(${SOURCE_DIR}/plot_sparse.py ${PATH_TO_BUILD} plot_sparse.py COPYONLY)

# Add subdirectory for linear_systems
add_subdirectory(../linear_systems linear_systems_build)
