include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc
    ${OpenCV_INCLUDE_DIRS} ${LIBXML2_INCLUDE_DIR})

link_directories(${OpenCV_LIB_DIR})
add_definitions(${LIBXML2_DEFINITIONS})

configure_file(config.h.in config.h ESCAPE_QUOTES)

cuda_add_library(morph morph.cu pyramid.cu upsample.cu render.cu
                       stencils.cpp cpuoptim.cpp imgio.cpp
                       downsample.cpp parameters.cpp)

target_link_libraries(morph util resample ${OpenCV_LIBS})

add_executable(gpumorph main.cpp param_io.cpp)

target_link_libraries(gpumorph morph ${LIBXML2_LIBRARIES})

set_target_properties(morph PROPERTIES FOLDER lib)
