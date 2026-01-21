cmake_minimum_required(VERSION 3.18)
project(moe_permute_sort CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set compute capability (modify as needed)
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

# Add executable
add_executable(permute_sort_test permute_sort_kernel.cu)

# Link CUDA runtime
target_link_libraries(permute_sort_test CUDA::cudart)

# Include CUB (part of CUDA Toolkit since CUDA 11)
target_include_directories(permute_sort_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
