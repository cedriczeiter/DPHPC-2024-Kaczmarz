## Description
This project implements a Kaczmarz method for solving systems of linear equations, with a simple PDE solver to demonstrate its functionality. The project has been organized to be built with CMake.

## How to Build and Run with CMake

### Steps to Build and Run

1. **Clone the Repository**:
   ```
   git clone <repository_url>
   cd parallel_kaczmarz_pde
   ```

2. **Create a Build Directory**:
   It is recommended to create a separate directory for the build files.
   ```
   mkdir build
   cd build
   ```

3. **Run CMake**:
   Generate the build configuration files.
   ```
   cmake ..
   ```

4. **Compile the Project**:
   Use the generated Makefiles to compile the project.
   ```
   make
   ```

5. **Run the Executable**:
   After a successful build, run the executable.
   ```
   ./parallel_kaczmarz_pde
   ```

### Clean Build
To clean the build, simply remove the `build` directory and recreate it, or use:
```
make clean
```
