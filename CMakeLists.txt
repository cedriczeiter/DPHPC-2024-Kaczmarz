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
    src/tests.c
    main.c
)

# Add executable target
add_executable(parallel_kaczmarz_pde ${SOURCES})

# Add tests
set(TESTSOURCES
    src/kaczmarz.c
    src/linear_system.c
)
set(TEST_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
enable_testing()

#add test for simple full rank matrix
add_executable(full_rank_test ${TESTSOURCES} tests/test_full_rank.c)
set_target_properties(full_rank_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIRECTORY})
add_test(
	NAME full_rank_test 
 	COMMAND full_rank_test
)

#add test for zero matrix
add_executable(zero_matrix_test ${TESTSOURCES} tests/test_zero_matrix.c)
set_target_properties(zero_matrix_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIRECTORY})
add_test(
	NAME zero_matrix_test
	COMMAND zero_matrix_test
)
