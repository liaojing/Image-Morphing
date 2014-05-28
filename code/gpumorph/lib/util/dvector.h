#ifndef ROD_DVECTOR_H
#define ROD_DVECTOR_H

#include <vector>
#include <climits>
#include "dalloc.h"
#include "hostdev.h"

namespace rod
{

template <class T>
class dvector_ptr
{
public:
    HOSTDEV dvector_ptr(T *data=NULL, size_t size=0);

    HOSTDEV void reset(T *data, size_t size);

    void fill(unsigned char byte);

    HOSTDEV const T &operator[](int idx) const;
#if __CUDA_ARCH__
    __device__ T &operator[](int idx);
#endif

    size_t copy_to_host(T *data, size_t s=UINT_MAX) const;
    size_t copy_from_host(const T *data, size_t s=UINT_MAX);

    HOSTDEV bool empty() const { return size()==0; }
    HOSTDEV size_t size() const { return m_size; }

    HOSTDEV T *data() { return m_data; }
    HOSTDEV const T *data() const { return m_data; }

    HOSTDEV T &back() const { return operator[](size()-1); }

    HOSTDEV operator T*() { return data(); }
    HOSTDEV operator const T*() const { return data(); }

    HOSTDEV const T *operator&() const { return m_data; }
    HOSTDEV T *operator&() { return m_data; }

    template <class U>
    HOSTDEV friend void swap(dvector_ptr<U> &a, dvector_ptr<U> &b);

private:
    T *m_data;
    size_t m_size;
};


template <class T>
class dvector
{
public:
    dvector(const T *data, size_t size);
    dvector(const dvector &that);
    dvector(size_t size=0);

    ~dvector();

    void resize(size_t size);

    void reset(T *data, size_t size);

    void clear() { m_size = 0; }
    void fill(unsigned char byte);

    const T &operator[](int idx) const;

    dvector &operator=(const dvector &that);

    void copy_to_host(std::vector<T> &out) const;
    void copy_from_host(const std::vector<T> &in);

    void copy_to_host(T *data, size_t s) const;
    void copy_from_host(const T *data, size_t s);

    void copy2D_to_host(T *out, size_t width, size_t height,
                        size_t rowstride) const;
    void copy2D_from_host(const T *in, size_t width, size_t height,
                          size_t rowstride);

    bool empty() const { return size()==0; }

    size_t size() const { return m_size; }

    T *data() { return m_data; }
    const T *data() const { return m_data; }

    T &back() const { return operator[](size()-1); }

    operator T*() { return data(); }
    operator const T*() const { return data(); }

    const T *operator&() const { return data(); }
    T *operator&() { return data(); }

    template <class U>
    friend void swap(dvector<U> &a, dvector<U> &b);

private:
    T *m_data;
    size_t m_size, m_capacity;
};

} // namespace rod

#include "dvector.hpp"

#endif
//vi: ai sw=4 ts=4
