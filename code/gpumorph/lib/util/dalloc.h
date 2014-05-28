#ifndef ROD_DALLOC_H
#define ROD_DALLOC_H

#include "error.h"

namespace rod
{

void *cuda_new(size_t elements, size_t elem_size);

template <class T>
T *cuda_new(size_t elements)
{
    return reinterpret_cast<T *>(cuda_new(elements, sizeof(T)));
}

void cuda_delete(void *ptr);

struct cuda_deleter
{
    template <class T>
    void operator()(T *ptr) const
    {
        cuda_delete(ptr);
    }
};

template <class T>
class cuda_allocator : public std::allocator<T>
{
public:
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    pointer allocate(size_type n, std::allocator<void>::const_pointer hint=0)
    {
        return cuda_new<T>(n);
    }

    void deallocate(pointer ptr, size_type n)
    {
        cuda_delete(ptr);
    }

    void construct(pointer ptr, const T &val)
    {
        // do nothing
    }
    void destroy(pointer ptr)
    {
        // do nothing
    }
};

} // namespace rod

#endif
