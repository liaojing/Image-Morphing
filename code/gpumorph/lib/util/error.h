#ifndef GPUMORPH_UTIL_ERROR_H
#define GPUMORPH_UTIL_ERROR_H

#include <string>

namespace rod
{

void check_cuda_error(const std::string &msg="");

}


#endif
