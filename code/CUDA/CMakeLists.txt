cmake_minimum_required(VERSION 2.8.0)

project(cg CXX)

find_package(CUDA REQUIRED)
if(UNIX)
    find_program(GCC NAMES gcc-4.5 gcc-4.5.4 gcc-4.5.3 gcc-4.5.2 gcc-4.5.1 gcc-4.5.0
        gcc-4.4 gcc-4.4.7 gcc-4.4.6 gcc-4.4.5 gcc-4.4.4 gcc-4.4.3 gcc-4.4.2
        gcc-4.4.1 gcc-4.4.0 gcc-4.3 gcc-4.3.6 gcc-4.3.5 gcc-4.3.4 gcc-4.3.3
        gcc-4.3.2 gcc-4.3.1 gcc-4.3.0 gcc)

    if(NOT GCC)
        message(FATAL_ERROR "Cannot find suitable compiler, must be gcc <= 4.5")
    endif()

    set(CUDA_HOST_COMPILER ${GCC})
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    #-Xptxas -dlcm=cg # gmem uses only L2 cache
    #-Xptxas -dlcm=ca # gmem uses both L1 and L2 cache (default)
    --ptxas-options=-v -lineinfo
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DNDEBUG -use_fast_math)
endif()


if(NOT CUDA_SM)
    set(CUDA_SM 30)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_${CUDA_SM},code=sm_${CUDA_SM})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)

cuda_add_library(cg cg.cu)
