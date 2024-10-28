## Description
This project implements a Kaczmarz method for solving systems of linear equations, with a simple PDE solver to demonstrate its functionality. The project has been organized to be built with CMake.

## Simple Setup to Build and Run

1. **Install Dependencies**
   Install CMake and Eigen if not already installed. 
   
   Install clang-format if you want to format code (using: `bash format_code.sh`).

2. **Clone the Repository**:
   ```
   git clone <repository_url>
   cd parallel_kaczmarz_pde
   ```

3. **Create a Build Directory**:
   It is recommended to create a separate directory for the build files.
   ```
   cd kaczmarz
   mkdir build
   cd build
   ```

4. **Run CMake**
   Generate the build configuration files.
   ```
   cmake ..
   ```

5. **Compile the Project**
   Use the generated Makefiles to compile the project.
   ```
   make
   ```

6. **Run Tests**:
   After a successful build, run the testing executable.
   ```
   ./testing
   ```

7. **Run the Example**:
   After a successful build, run the example executable.
   ```
   ./kaczmarz_example
   ```
   
8. **Run the Benchmarking**:
   After a successful build, run the benchmark executable.
   ```
   ./benchmark_executable
   ```

 9. **Plot the Benchmarking**:
   After a successful build and benchmarking, plot the results of the benchmark.
   ```
   python plot.py
   ```

10. **Clean the Build If Necessary**
   To clean the build, simply remove the `build` directory and recreate it, or use:
   ```
   make clean
   ```

## Project Structure
On the top level, the project is divided into three parts:
  * `linear_systems` contains a simple (statically-linked) library for the representation of linear systems. This library is shared between the remaining two parts...
  * `mesh_builder` uses LehrFEM++ to generate examples of linear systems that correspond to some boundary value problems.
  * `kaczmarz` is the heart of the project. It consist of our implementation of the Kaczmarz solver for solving LSEs.

### Building Individual Parts
Each of three parts can be built independently. Analogously to the setup instructions above, the recommended way is to create a `<part>/build` folder and configure CMake inside of it. In case of `mesh_builder` and `kaczmarz`, the `linear_systems` library will be automatically built during the build of the depedent part. There is no need to build `linear_systems` separately for that.

### Helpful Scripts
This directory contains the following scripts that might come in handy:
* `clear_all_build.sh` deletes any existing `build` folders that were created at the recommended paths.
* `build_all_anew.sh` runs `clear_all_build.sh` and then creates a build folder for each part at the recommended path and performs the build.
* `format_code.sh` formats all source and header files using `clang-format`. This should be run before making a PR to merge to the main branch. We have a CI check that will prevent merging unformatted code to main.

### Other Files
* The `generated_bvp_matrices` folder contains sample LSEs created from BVPs using the mesh builder. They are meant to be used a inputs for our Kaczmarz solver implementation.
* `sources.txt` lists we sources that we've used throughout the project.
