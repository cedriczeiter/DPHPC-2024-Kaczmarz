cmake_minimum_required(VERSION 3.10)
project(parallel_kaczmarz_pde)

# Set C standard
set(CMAKE_C_STANDARD 11)

# Enable all warnings and extra warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pthread")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/linear_system.c
    src/kaczmarz.c
    #src/pde_solver.c
    src/tests.c
    main.c
)

# Add executable target
add_executable(parallel_kaczmarz_pde ${SOURCES})
