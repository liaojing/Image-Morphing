#include <cuda_runtime.h>
#include <algorithm>
namespace rod
{

// class dvector<T> --------------------------------------------------------

template <class T>
dvector<T>::dvector(const T *data, size_t size)
    : m_size(0), m_capacity(0), m_data(0)
{
    resize(size);
    cudaMemcpy(this->data(), const_cast<T *>(data), size*sizeof(T),
               cudaMemcpyHostToDevice);
    check_cuda_error("Error during memcpy from host to device");
}

template <class T>
dvector<T>::dvector(const dvector &that)
    : m_size(0), m_capacity(0), m_data(0)
{
    *this = that;
}

template <class T>
dvector<T>::dvector(size_t size)
    : m_size(0), m_capacity(0), m_data(0)
{
    resize(size);
}

template <class T>
dvector<T>::~dvector()
{
    cuda_delete(m_data);
    m_data = 0;
    m_capacity = 0;
    m_size = 0;
}

template <class T>
void dvector<T>::resize(size_t size)
{
    if(size > m_capacity)
    {
        cuda_delete(m_data);
        m_data = 0;
        m_capacity = 0;
        m_size = 0;

        m_data = cuda_new<T>(size);
        m_capacity = size;
        m_size = size;
    }
    else
        m_size = size;
}

template <class T>
void dvector<T>::reset(T *data, size_t size)
{
    cuda_delete(m_data);
    m_data = NULL;

    m_capacity = m_size = size;
    m_data = data;
}

template <class T>
void dvector<T>::fill(unsigned char byte)
{
    cudaMemset(m_data, byte, m_size*sizeof(T));
}

template <class T>
const T &dvector<T>::operator[](int idx) const
{
    static T value;
    cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

template <class T>
dvector<T> &dvector<T>::operator=(const dvector &that)
{
    resize(that.size());
    cudaMemcpy(data(), that.data(), size()*sizeof(T), cudaMemcpyDeviceToDevice);
    check_cuda_error("Error during memcpy from device to device");
    return *this;
}

template <class T>
void dvector<T>::copy_to_host(T *data, size_t s) const
{
    using std::min;

    cudaMemcpy(data, this->data(), min(size(),s)*sizeof(T),
               cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");
}

template <class T>
void dvector<T>::copy_from_host(const T *data, size_t s)
{
    using std::min;

    resize(s);

    cudaMemcpy(this->data(), data, s*sizeof(T), cudaMemcpyHostToDevice);
    check_cuda_error("Error during memcpy from device to host");
}

namespace detail
{
    // workaround due to std::vector<bool> being a completely different beast

    template <class T>
    void copy_to_host(std::vector<T> &out, const dvector<T> &in)
    {
        out.resize(in.size());
        in.copy_to_host(&out[0], out.size());
    }
    inline void copy_to_host(std::vector<bool> &out, const dvector<bool> &in)
    {
        bool *temp = NULL;
        try
        {
            temp = new bool[in.size()];

            in.copy_to_host(temp, in.size());

            out.assign(temp, temp + in.size());
            delete[] temp;
        }
        catch(...)
        {
            delete[] temp;
            throw;
        }
    }

    template <class T>
    void copy_from_host(dvector<T> &out, const std::vector<T> &in)
    {
        out.resize(in.size());
        out.copy_from_host(&in[0], in.size());
    }

    inline void copy_from_host(dvector<bool> &out, const std::vector<bool> &in)
    {
        std::vector<unsigned char> temp;
        temp.assign(in.begin(), in.end());

        out.copy_from_host((const bool *)&temp[0], temp.size());
    }

}

template <class T>
void dvector<T>::copy_to_host(std::vector<T> &out) const
{
    detail::copy_to_host(out, *this);
}

template <class T>
void dvector<T>::copy_from_host(const std::vector<T> &in)
{
    detail::copy_from_host(*this, in);
}

template <class T>
void dvector<T>::copy2D_to_host(T *out, size_t width, size_t height,
                                size_t rowstride) const
{
    cudaMemcpy2D(out, width*sizeof(T),
                 data(), rowstride*sizeof(T),
                 width*sizeof(T), height, cudaMemcpyDeviceToHost);

    check_cuda_error("Error during memcpy from device to host");
}

template <class T>
void dvector<T>::copy2D_from_host(const T *in, size_t width, size_t height,
                                  size_t rowstride)
{
    resize(rowstride*height);

    cudaMemcpy2D(data(), rowstride*sizeof(T), in, width*sizeof(T),
                 width*sizeof(T), height, cudaMemcpyHostToDevice);

    check_cuda_error("Error during memcpy from host to device");
}

template <class T>
void swap(dvector<T> &a, dvector<T> &b)
{
    using std::swap;
    std::swap(a.m_data, b.m_data);
    swap(a.m_size, b.m_size);
    swap(a.m_capacity, b.m_capacity);
}


template <class T>
std::vector<T> to_host(const T *d_vec, unsigned len)
{
    std::vector<T> out;
    out.resize(len);

    cudaMemcpy(&out[0], d_vec, len*sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");

    return out;
}

template <class T>
std::vector<T> to_host(const dvector<T> &v)
{
    return to_cpu(v.data(), v.size());
}

// class dvector_ptr<T> -------------------------------------------------------

template <class T>
HOSTDEV
dvector_ptr<T>::dvector_ptr(T *data, size_t size)
    : m_size(size), m_data(data)
{
}

template <class T>
HOSTDEV
void dvector_ptr<T>::reset(T *data, size_t size)
{
    m_size = size;
    m_data = data;
}

template <class T>
void dvector_ptr<T>::fill(unsigned char byte)
{
    cudaMemset(m_data, byte, m_size*sizeof(T));
}

#if !__CUDA_ARCH__
template <class T>
const T &dvector_ptr<T>::operator[](int idx) const
{
    static T value;
    cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}
#else
template <class T>
const T &dvector_ptr<T>::operator[](int idx) const
{
    return data()[idx];
}
template <class T>
__device__ T &dvector_ptr<T>::operator[](int idx)
{
    return data()[idx];
}
#endif


template <class T>
size_t dvector_ptr<T>::copy_to_host(T *data, size_t s) const
{
    using std::min;

    size_t nwritten = min(size(),s);

    cudaMemcpy(data, this->data(), nwritten*sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");
    return nwritten;
}

template <class T>
size_t dvector_ptr<T>::copy_from_host(const T *data, size_t s)
{
    using std::min;

    size_t nwritten = min(size(),s);

    cudaMemcpy(this->data(), data, nwritten*sizeof(T), cudaMemcpyHostToDevice);
    check_cuda_error("Error during memcpy from device to host");
    return nwritten;
}

template <class T>
HOSTDEV
void swap(dvector_ptr<T> &a, dvector_ptr<T> &b)
{
    using std::swap;

    swap(a.m_data, b.m_data);
    swap(a.m_size, b.m_size);
}

} // namespace rod
