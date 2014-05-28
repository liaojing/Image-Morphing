include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)

find_package(PkgConfig)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(GSL gsl)

    if(GSL_FOUND)
        set(HAS_GSL 1)
    endif()
endif()

configure_file(config.h.in config.h ESCAPE_QUOTES)

cuda_add_library(util error.cpp timer.cpp recfilter.cu dalloc.cpp
                 dimage.cpp imgop_basic.cu imgop_conv.cu
                 imgop_gaussblur.cpp imgop_upsample.cu imgop_downsample.cu) 

set_target_properties(util PROPERTIES FOLDER lib)
