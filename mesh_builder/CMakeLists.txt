cmake_minimum_required(VERSION 3.10)

# Without setting this, CMake gives a warning
# Setting this policy ensures whenever some content is fetched, it will get a fresh timestamp so
# that CMake doesn't skip rebuilding any of its dependents
cmake_policy(SET CMP0135 NEW)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.3.tar.gz"
    SHA1 "0dfbc2cb5c4cf7e83533733bdfd2125ff96680cb"
    LOCAL
)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


### Setup CMake Project ###
project(MeshBuilder)


#setup compiler
set(CMAKE_CXX_STANDARD 20)

#enable language support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Enable all warnings and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")


## Get Eigen
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# Get LehrFEMpp
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(../linear_systems linear_systems_build)


# define executable
add_executable(mesh_builder ${SOURCE_DIR}/main.cpp)

target_link_libraries(mesh_builder nlohmann_json::nlohmann_json) 
target_link_libraries(mesh_builder Eigen3::Eigen) 
target_link_libraries(mesh_builder LF::lf.io)
target_link_libraries(mesh_builder LF::lf.refinement)
target_link_libraries(mesh_builder LF::lf.mesh.hybrid2d)
target_link_libraries(mesh_builder LF::lf.mesh)
target_link_libraries(mesh_builder LF::lf.mesh.utils)
target_link_libraries(mesh_builder LF::lf.mesh.test_utils)
target_link_libraries(mesh_builder LF::lf.fe)
target_link_libraries(mesh_builder LF::lf.assemble)
target_link_libraries(mesh_builder LF::lf.uscalfe)
target_link_libraries(mesh_builder linear_systems)

