./clear_all_builds.sh

build_directory() {
    local dir_name=$1
    mkdir -p "$dir_name/build" || exit
    cd "$dir_name/build" || exit
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j
    cd ../..
}

build_directory "linear_systems"
build_directory "kaczmarz"
build_directory "mesh_builder"
