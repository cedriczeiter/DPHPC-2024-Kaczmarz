## Description
This project implements a Kaczmarz method for solving systems of linear equations, with a simple PDE solver to demonstrate its functionality. The project has been organized to be built with CMake.

## How to Build and Run with CMake

### Steps to Build and Run

1. **Install Dependencies**
   Install CMake and Eigen if not already installed.

2. **Clone the Repository**:
   ```
   git clone <repository_url>
   cd parallel_kaczmarz_pde
   ```

3. **Create a Build Directory**:
   It is recommended to create a separate directory for the build files.
   ```
   mkdir build
   cd build
   ```

4. **Run CMake**:
   Generate the build configuration files.
   ```
   cmake ..
   ```

5. **Compile the Project**:
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

### Clean Build
To clean the build, simply remove the `build` directory and recreate it, or use:
```
make clean
```
