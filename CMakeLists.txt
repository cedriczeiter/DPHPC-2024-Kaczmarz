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

add_executable(kaczmarz_example ${SOURCE_DIR}/main.cpp ${SOURCE_DIR}/random_dense_system/random_dense_system.cpp ${SOURCE_DIR}/kaczmarz_serial/kaczmarz.c)
#add_subdirectory(src/random_dense_system)
target_include_directories(kaczmarz_example PRIVATE ${SOURCE_DIR}/random_dense_system ${SOURCE_DIR}/kaczmarz_serial)
target_link_libraries(kaczmarz_example Eigen3::Eigen)
