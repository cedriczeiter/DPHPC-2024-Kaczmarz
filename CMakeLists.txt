cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
#set Cpp standard
set(CMAKE_CXX_STANDARD 11)

# Enable all warnings and extra warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pthread")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SOURCES
  ${SOURCE_DIR}/random_dense_system/random_dense_system.cpp
  ${SOURCE_DIR}/kaczmarz_serial/kaczmarz.c
)

include(ExternalProject)
ExternalProject_Add(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           main
    SOURCE_DIR        "${CMAKE_BINARY_DIR}/googletest-src"
    BINARY_DIR        "${CMAKE_BINARY_DIR}/googletest-build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build"
)

add_executable(kaczmarz_example ${SOURCE_DIR}/kaczmarz_example.cpp ${SOURCES})
add_executable(testing ${SOURCE_DIR}/testing.cpp ${SOURCES})

target_include_directories(kaczmarz_example PRIVATE ${SOURCE_DIR}/random_dense_system ${SOURCE_DIR}/kaczmarz_serial)
target_include_directories(testing PRIVATE ${SOURCE_DIR}/random_dense_system ${SOURCE_DIR}/kaczmarz_serial)

target_link_libraries(kaczmarz_example Eigen3::Eigen)
target_link_libraries(testing Eigen3::Eigen)
target_link_libraries(testing gtest)
