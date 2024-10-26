#!/bin/bash

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo "clang-format could not be found. Please install clang-format to proceed."
    exit 1
fi

# Define the formatting style (Google style is used as an example)
STYLE="Google"

# Find and format all .cpp and .hpp files
echo "Formatting all C++ files in the project using clang-format with style: $STYLE"
find . -iname '*.hpp' -o -iname '*.cpp' | xargs clang-format -i --style=$STYLE

echo "Formatting complete."
