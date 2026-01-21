cmake_minimum_required(VERSION 3.18)
project(moe_fused_gemm_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# SM80 only.
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

set(FUSED_MOE_SOURCES
    cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.cu
)

add_library(fused_moe_gemm STATIC ${FUSED_MOE_SOURCES})

target_include_directories(fused_moe_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
)

target_compile_options(fused_moe_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(fused_moe_gemm
    CUDA::cudart
)

add_executable(test_moe_fused_gemm test/test_moe_fused_gemm.cu)
target_link_libraries(test_moe_fused_gemm
    fused_moe_gemm
    CUDA::cudart
)

target_compile_options(test_moe_fused_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
