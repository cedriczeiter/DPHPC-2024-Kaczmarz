cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
#setup CXX standard 
set(CMAKE_CXX_STANDARD 20)

# Enable all warnings and extra warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pthread")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SOURCES
  ${SOURCE_DIR}/kaczmarz_serial/kaczmarz.cpp
  ${SOURCE_DIR}/kaczmarz_serial/random_kaczmarz.cpp
)

# Fetch GoogleTest from the official GitHub repository
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main # Use the latest stable version tag, or you can use "main" for the latest commit
)

# This will make sure the content is downloaded and added
FetchContent_MakeAvailable(googletest)

add_executable(benchmark_executable ${SOURCE_DIR}/benchmarking.cpp ${SOURCES})
add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp ${SOURCES})
add_executable(testing ${SOURCE_DIR}/testing.cpp ${SOURCES})

target_include_directories(kaczmarz_example PRIVATE ${SOURCE_DIR}/linear_systems ${SOURCE_DIR}/kaczmarz_serial)
target_include_directories(testing PRIVATE ${SOURCE_DIR}/linear_systems ${SOURCE_DIR}/kaczmarz_serial)
target_include_directories(benchmark_executable PRIVATE ${SOURCE_DIR}/linear_systems ${SOURCE_DIR}/kaczmarz_serial)


CONFIGURE_FILE(${SOURCE_DIR}/plot.py ${PATH_TO_BUILD} plot.py COPYONLY)

add_subdirectory(../linear_systems linear_systems_build)

target_link_libraries(kaczmarz_example PRIVATE Eigen3::Eigen linear_systems)
target_link_libraries(testing PRIVATE Eigen3::Eigen linear_systems gtest)
target_link_libraries(benchmark_executable PRIVATE Eigen3::Eigen linear_systems)
