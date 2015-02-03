include_directories(${CUDA_TOOLKIT_INCLUDE})
add_library(resample error.cpp chronos.cpp discrete.cpp dlti.cpp image.cpp
                     util.cpp scale.cpp)
target_link_libraries(resample util)
