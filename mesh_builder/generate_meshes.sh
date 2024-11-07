#!/bin/bash

#build project
export CC=gcc-13 #rempve these exports if they do not work on your machine
export CXX=g++-13
cmake -B build
cd build/
make
cd ../..

#create meshes for problem 1
DIM=2
PROBLEM=1
SELECTOR=0
SCALE=1
DEGREE=2
REFINEMENT=1
echo "Starting mesh generation"
for i in $(seq 1 10);
do
    echo \
"{
    \"dimension\": $DIM,
    \"problem\": $PROBLEM,
    \"selector\": $SELECTOR,
    \"scale\": $SCALE,
    \"degree\": $DEGREE,
    \"refinement\": $REFINEMENT
}" \
    > mesh_builder/configuration.json
    ./mesh_builder/build/mesh_builder mesh_builder/configuration.json > generated_bvp_matrices/problem${PROBLEM}_complexity${i}.txt
    #DEGREE=$((DEGREE+1)) #we could also increase the degree, this is commentd out right now to have somewhat smaller linear systems
    REFINEMENT=$((REFINEMENT+1))
    echo "Finished mesh generation for mesh nr ${i}"
done
