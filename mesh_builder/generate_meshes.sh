#!/bin/bash

#build project
export CC=gcc-13 #remove these exports if they do not work on your machine
export CXX=g++-13
cmake -B build
cd build/
make
cd ../..

#create meshes for problems 1, 2, 3
for j in $(seq 1 3);
do
    for k in $(seq 1 3); #for degrees 1 & 2
    do
        DIM=2
        PROBLEM=$j
        SELECTOR=0
        SCALE=1
        DEGREE=$k
        REFINEMENT=1
        echo "Starting mesh generation"
        for i in $(seq 1 6); #up to complexity 6 atm
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
            ./mesh_builder/build/mesh_builder mesh_builder/configuration.json > generated_bvp_matrices/problem${PROBLEM}_complexity${i}_degree${DEGREE}.txt
            #DEGREE=$((DEGREE+1)) #we could also increase the degree, this is commentd out right now to have somewhat smaller linear systems
            REFINEMENT=$((REFINEMENT+1))
            echo "Finished mesh generation for mesh nr ${i}"
        done
    done

done
