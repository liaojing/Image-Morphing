#include <stdexcept>

namespace rod
{

// class dimage -----------------------------------------------------------

template <class T, int C>
dimage<T,C>::dimage(texel_type *data, int width, int height, int rowstride)
    : m_data(data, rowstride*height)
    , m_width(width)
    , m_height(height)
    , m_rowstride(rowstride)
{
}

template <class T, int C>
dimage<T,C>::dimage(int width, int height, int rowstride)
{
    resize(width, height, rowstride);
}

template <class T, int C>
dimage<T,C>::dimage() : m_width(0), m_height(0), m_rowstride(0)
{
}

template <class T, int C>
void dimage<T,C>::reset(texel_type *data, int width, int height, int rowstride)
{
    m_width = width;
    m_height = height;
    if(rowstride == 0)
        m_rowstride = ((m_width + 256-1)/256)*256; // multiple of 256
    else
    {
        m_rowstride = rowstride;
        if(rowstride < width)
            throw std::runtime_error("Bad row stride");
    }

    m_data.reset(data, m_rowstride*m_height*C);
}

template <class T, int C>
void dimage<T,C>::copy_to_host(mutable_texel_type *out, int rowstride) const
{
    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(out+rowstride*height()*i, rowstride*sizeof(texel_type),
                     &m_data+i*channelstride(),
                     this->rowstride()*sizeof(texel_type),
                     width()*sizeof(texel_type), height(),
                     cudaMemcpyDeviceToHost);
    }

    check_cuda_error("Error during memcpy from device to host");
}

template <class T, int C>
void dimage<T,C>::copy_to_host(std::vector<mutable_texel_type> &out, int rowstride) const
{
    if(rowstride == 0)
        rowstride = width();

    out.resize(rowstride*height());
    copy_to_host(&out[0],rowstride);
}

template <class T, int C>
void dimage<T,C>::copy_from_host(const texel_type *in, int w, int h, int rs)
{
    resize(w, h);

    if(rs == 0)
        rs = w;

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(&m_data+i*channelstride(), rowstride()*sizeof(texel_type),
                     in+i*rs*height(), rs*sizeof(texel_type),
                     width()*sizeof(texel_type), height(), cudaMemcpyHostToDevice);
    }

    check_cuda_error("Error during memcpy from host to device");
}

template <class T, int C>
void dimage<T,C>::copy_from_host(const std::vector<mutable_texel_type> &in,
                    int width, int height, int rowstride)
{
    assert(in.size() == (rowstride != 0 ? rowstride : width)*height);
    copy_from_host(&in[0], width, height, rowstride);
}

template <class T, int C>
void dimage<T,C>::resize(int width, int height, int rowstride)
{
    m_width = width;
    m_height = height;
    if(rowstride == 0)
        m_rowstride = ((m_width + 256-1)/256)*256; // multiple of 256
    else
    {
        if(rowstride < m_width)
            throw std::runtime_error("Bad row stride");
        m_rowstride = rowstride;
    }

    m_data.resize(m_rowstride*m_height*C);

}

template <class T, int C>
dimage<T,C> &dimage<T,C>::operator=(const dimage_ptr<const texel_type, C> &img)
{
    resize(img.width(), img.height(), img.rowstride());

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(&m_data+i*channelstride(),rowstride()*sizeof(texel_type),
                     &img+i*img.channelstride(), rowstride()*sizeof(texel_type),
                     width()*sizeof(texel_type), height(), cudaMemcpyDeviceToDevice);
    }

    return *this;
}

template <class T, int C>
dimage<T,C> &dimage<T,C>::operator=(const dimage &that)
{
    m_data = that.m_data;

    m_width = that.m_width;
    m_height = that.m_height;
    m_rowstride = that.m_rowstride;
    return *this;
}

// class dimage_ptr ---------------------------------------------------------

template <class T, int C>
HOSTDEV
dimage_ptr<T,C>::dimage_ptr(texel_type *data, int width, int height,
                            int rowstride)
    : m_width(width), m_height(height), m_rowstride(rowstride)
    , m_data(data)
{
}

template <class T, int C>
template <class P>
HOSTDEV
dimage_ptr<T,C>::dimage_ptr(const dimage_ptr<P,C> &that,
           typename enable_if<is_convertible<P,T>::value>::type*)
    : m_width(that.m_width), m_height(that.m_height)
    , m_rowstride(that.m_rowstride)
    , m_data(that.m_data)
{
}


template <class T, int C>
void dimage_ptr<T,C>::copy_to_host(mutable_texel_type *out, int rowstride) const
{
    if(rowstride == 0)
        rowstride = width();

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(out+rowstride*height()*i, rowstride*sizeof(texel_type),
                     m_data+i*channelstride(),
                     this->rowstride()*sizeof(texel_type),
                     width()*sizeof(texel_type), height(),
                     cudaMemcpyDeviceToHost);
    }

    check_cuda_error("Error during memcpy from device to host");
}

template <class T, int C>
void dimage_ptr<T,C>::copy_to_host(std::vector<mutable_texel_type> &out, int rowstride) const
{
    if(rowstride == 0)
        rowstride = width();

    out.resize(rowstride*height());
    copy_to_host(&out[0]);
}

template <class T, int C>
void dimage_ptr<T,C>::copy_from_host(const texel_type *in, int w, int h, int rs)
{
    if(w != width() || h!=height())
       throw std::runtime_error("Image dimensions mismatch");

    if(rs == 0)
        rs = w;

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(m_data+i*channelstride(), rowstride()*sizeof(texel_type),
                     in+i*rs*height(), rs*sizeof(texel_type),
                     width()*sizeof(texel_type), height(), cudaMemcpyHostToDevice);
    }

    check_cuda_error("Error during memcpy from host to device");
}

template <class T, int C>
void dimage_ptr<T,C>::copy_from_host(const std::vector<mutable_texel_type> &in,
                    int width, int height, int rowstride)
{
    assert(in.size() == (rowstride != 0 ? rowstride : width)*height);
    copy_from_host(&in[0], width, height, rowstride);
}

template <class T, int C>
dimage_ptr<T,C> &dimage_ptr<T,C>::operator=(dimage_ptr<const T,C> img)
{
    if(width() != img.width() || height() != img.height())
        throw std::runtime_error("Image dimensions don't match");

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D((*this)[i], rowstride()*sizeof(texel_type),
                     img[i], img.rowstride()*sizeof(texel_type),
                     width()*sizeof(texel_type), height(),
                     cudaMemcpyDeviceToDevice);
    }
    return *this;
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,1> dimage_ptr<T,C>::operator[](int i)
{
    return dimage_ptr<T,1>(&m_data[i*channelstride()],
                           width(), height(), rowstride());
}

template <class T, int C>
HOSTDEV
dimage_ptr<const T,1> dimage_ptr<T,C>::operator[](int i) const
{
    return dimage_ptr<const T,1>(&m_data[i*channelstride()],
                                 width(), height(), rowstride());
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,C> &dimage_ptr<T,C>::operator++()
{
    ++m_data;
    return *this;
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,C> dimage_ptr<T,C>::operator++(int)
{
    dimage_ptr ret(*this);
    ++*this;
    return ret;
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,C> &dimage_ptr<T,C>::operator--()
{
    --m_data;
    return *this;
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,C> dimage_ptr<T,C>::operator--(int)
{
    dimage_ptr ret(*this);
    --*this;
    return ret;
}

template <class T, int C>
HOSTDEV
dimage_ptr<T,C> &dimage_ptr<T,C>::operator+=(int off)
{
    m_data += off;
    return *this;
}

template <class T, int C>
dimage_ptr<T,C> dimage<T,C>::operator&()
{
    return dimage_ptr<T,C>(m_data.data(), width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<const T,C> dimage<T,C>::operator&() const
{
    return dimage_ptr<const T,C>(m_data.data(), width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<T,1> dimage<T,C>::operator[](int i)
{
    return dimage_ptr<T,1>(m_data.data()+i*channelstride(),
                           width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<const T, 1> dimage<T,C>::operator[](int i) const
{
    return dimage_ptr<const T,1>(m_data.data()+i*channelstride(),
                                 width(), height(), rowstride());
}

// free functions -----------------------------------------------------------

template <class T, int C>
void subimage(dimage<T,C> &dst, dimage_ptr<T,C> src, int x, int y, int w, int h)
{
    typedef typename dimage<T,C>::texel_type texel_type;

    if(x >= src.width() || x < 0)
        throw std::invalid_argument("Bad x position");

    if(y >= src.height() || y < 0)
        throw std::invalid_argument("Bad x position");

    w = min(w, src.width()-x);
    h = min(w, src.height()-h);

    dst.resize(w,h);
    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(&dst[i], dst.rowstride()*sizeof(texel_type),
                     &src[i]+y*src.rowstride()+x, src.rowstride()*sizeof(texel_type),
                     w*sizeof(texel_type), h, cudaMemcpyDeviceToDevice);
    }
}

template <class T, int C>
void subimage(dimage_ptr<T,C> dst, dimage_ptr<T,C> src, int x, int y, int w, int h)
{
    typedef typename dimage<T,C>::texel_type texel_type;

    if(dst.width() != w || dst.height()!=h)
        throw std::invalid_argument("Bad destination image dimensions");

    if(x >= src.width() || x < 0)
        throw std::invalid_argument("Bad x position");

    if(y >= src.height() || y < 0)
        throw std::invalid_argument("Bad x position");

    w = min(w, src.width()-x);
    h = min(w, src.height()-h);

    for(int i=0; i<C; ++i)
    {
        cudaMemcpy2D(dst[i], dst.rowstride()*sizeof(texel_type),
                     &src[i]+y*src.rowstride()+x, src.rowstride()*sizeof(texel_type),
                     w*sizeof(texel_type), h, cudaMemcpyDeviceToDevice);
    }
}

} // namespace rod
