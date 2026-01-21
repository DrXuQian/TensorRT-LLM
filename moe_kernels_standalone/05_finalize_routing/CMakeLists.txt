cmake_minimum_required(VERSION 3.18)
project(moe_finalize_routing CUDA CXX)

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
add_executable(finalize_routing_test finalize_routing_kernel.cu)

# Link CUDA runtime
target_link_libraries(finalize_routing_test CUDA::cudart)

# Include directories
target_include_directories(finalize_routing_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
