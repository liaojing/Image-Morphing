#include <cuda_runtime.h>
#include <stdexcept>
#include "error.h"

namespace rod
{

void *cuda_new(size_t elements, size_t elem_size)
{
    void *ptr = NULL;

    cudaMalloc(&ptr, elements*elem_size);
    check_cuda_error("Memory allocation error");
    if(ptr == NULL)
        throw std::runtime_error("Memory allocation error");

    return ptr;
}

void cuda_delete(void *ptr)
{
    cudaFree(ptr);
    check_cuda_error("Error freeing memory");
}

}
