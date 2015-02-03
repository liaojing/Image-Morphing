cmake_minimum_required(VERSION 2.8.0)

project(gpumorph CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/etc)

if(NOT WIN32)
    include(PCHSupport)
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(UNIX)
    find_program(GCC NAMES gcc-4.6 gcc-4.6.2 gcc-4.5 gcc-4.5.4 gcc-4.5.3 gcc-4.5.2 gcc-4.5.1 gcc-4.5.0
        gcc-4.4 gcc-4.4.7 gcc-4.4.6 gcc-4.4.5 gcc-4.4.4 gcc-4.4.3 gcc-4.4.2
        gcc-4.4.1 gcc-4.4.0 gcc-4.3 gcc-4.3.6 gcc-4.3.5 gcc-4.3.4 gcc-4.3.3
        gcc-4.3.2 gcc-4.3.1 gcc-4.3.0 gcc)

    if(NOT GCC)
        message(FATAL_ERROR "Cannot find suitable compiler, must be gcc <= 4.6")
    endif()

    set(CUDA_HOST_COMPILER ${GCC})
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    #-Xptxas -dlcm=cg # gmem uses only L2 cache
    #-Xptxas -dlcm=ca # gmem uses both L1 and L2 cache (default)
    --ptxas-options=-v -lineinfo
)

set(CMAKE_CXX_FLAGS -mno-sse2)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DNDEBUG -use_fast_math -g)
endif()


if(NOT CUDA_SM)
    set(CUDA_SM 30)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_${CUDA_SM},code=sm_${CUDA_SM})

include_directories(lib)

add_subdirectory(lib EXCLUDE_FROM_ALL)
add_subdirectory(src)
