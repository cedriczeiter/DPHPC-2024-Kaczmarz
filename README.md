## Description
This project implements a Kaczmarz method for solving systems of linear equations, with a simple PDE solver to demonstrate its functionality. The project has been organized to be built with CMake.

## How to Build and Run with CMake

### Steps to Build and Run

1. **Install Dependencies**
   Install CMake and Eigen if not already installed. 
   
   Install clang-format if you want to format code (using: bash format_code.sh).

2. **Clone the Repository**:
   ```
   git clone <repository_url>
   cd parallel_kaczmarz_pde
   ```

3. **Run Build Script**:
   ```
   bash build_all_anew.sh
   ```
   This creates build folders in all the main folders. In there one can find already created executables. (No need to run make-command anymore)



------------------Needs to be updated from here on---------------------------------------

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
   After a successful build and benchmarking plot the results of the benchmark.
   ```
   python plot.py
   ```
### Clean Build
To clean the build, simply remove the `build` directory and recreate it, or use:
```
make clean
```
