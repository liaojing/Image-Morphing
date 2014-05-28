#ifndef NLFILTER_DIMAGE_H
#define NLFILTER_DIMAGE_H

#include <cassert>
#include "dvector.h"
#include "util.h"
#include "pixel_traits.h"
#include "dimage_fwd.h"

namespace rod
{

template <class T, int C>
class dimage
{
public:
    typedef typename pixel_traits<T,1>::texel_type texel_type;
    typedef typename remove_const<texel_type>::type mutable_texel_type;

    dimage(texel_type *data, int width, int height, int rowstride);
    dimage(int width, int height, int rowstride=0);
    dimage();

    void reset(texel_type *data, int width, int height, int rowstride);
    void reset() { reset(NULL,0,0,0); }

    friend void set_zero(dimage &img) { img.m_data.fillzero(); }

    void copy_to_host(mutable_texel_type *out, int rowstride=0) const;
    void copy_to_host(std::vector<mutable_texel_type> &out, int rowstride=0) const;

    void copy_from_host(const texel_type *in, int w, int h, int rs=0);
    void copy_from_host(const std::vector<mutable_texel_type> &in,
                        int width, int height, int rowstride=0);

    void resize(int width, int height, int rowstride=0);

    bool empty() const { return m_data.empty(); }

    dimage &operator=(const dimage_ptr<const texel_type, C> &img);
    dimage &operator=(const dimage &that);

    int width() const { return m_width; }
    int height() const { return m_height; }
    int rowstride() const { return m_rowstride; }
    int channelstride() const { return rowstride()*height(); }

    dimage_ptr<T,C> operator&();
    dimage_ptr<const T,C> operator&() const;

    operator texel_type*() { return m_data; }
    operator const texel_type*() const { return m_data; }

    int offset_at(int x, int y) const { return y*rowstride()+x; }
    int offset_at(int2 pos) const { return offset_at(pos.x, pos.y); }
    bool is_inside(int x, int y) const { return x < width() && y < height(); }
    bool is_inside(int2 pos) const { return is_inside(pos.x, pos.y); }

    dimage_ptr<T,1> operator[](int i);

    dimage_ptr<const T, 1> operator[](int i) const;

    friend void swap(dimage &orig, dimage &dest)
    {
        using std::swap;

        swap(orig.m_data, dest.m_data);
        swap(orig.m_width, dest.m_width);
        swap(orig.m_height, dest.m_height);
        swap(orig.m_rowstride, dest.m_rowstride);
    }

private:
    dvector<texel_type> m_data;
    int m_width, m_height, m_rowstride;
};

template <class T, int C>
class dimage_ptr
{
public:
    // these should be defined in dimage.hpp but nvcc (cuda 4.2) doesn't
    // like outside definition of nested template class of a template class
    template <int D, class EN=void>
    class pixel_proxy/*{{{*/
    {
        dimage_ptr &m_img;

    public:
        HOSTDEV
        pixel_proxy(dimage_ptr &img): m_img(img) {}

        typedef typename pixel_traits<T,C>::pixel_type value_type;

        template <class V>
        HOSTDEV
        pixel_proxy &operator=(const V &p)
        {
            pixel_traits<texel_type,D>::assign(m_img.m_data,
                                               m_img.channelstride(),
                                           pixel_traits<T,C>::make_pixel(p));
            return *this;
        }

        HOSTDEV
        pixel_proxy &operator+=(const value_type &v)
        {
            value_type temp;
            pixel_traits<texel_type,D>::assign(temp, m_img.m_data, m_img.channelstride());
            temp += v;
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), temp);
            return *this;
        }

        HOSTDEV
        pixel_proxy &operator-=(const value_type &v)
        {
            value_type temp;
            pixel_traits<texel_type,D>::assign(temp, m_img.m_data, m_img.channelstride());
            temp -= v;
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), temp);
            return *this;
        }

        HOSTDEV
        operator value_type() const
        {
            value_type val;
            pixel_traits<texel_type,D>::assign(val, m_img.m_data, m_img.channelstride());
            return val;
        }

        T *operator&() { return m_img.m_data; }
        const T *operator&() const { return m_img.m_data; }
    };/*}}}*/

    template <int D>
    class pixel_proxy<D, typename enable_if<(D<=0 || D>=5)>::type>/*{{{*/
    {
    public:
        HOSTDEV pixel_proxy(dimage_ptr<T,C> &img) {}
    };/*}}}*/

    template <int D, class EN=void>
    class const_pixel_proxy/*{{{*/
    {
        const dimage_ptr &m_img;

    public:
        HOSTDEV const_pixel_proxy(const dimage_ptr &img) : m_img(img) {}

        typedef typename pixel_traits<T,C>::pixel_type value_type;

        HOSTDEV operator value_type() const
        {
            value_type val;
            pixel_traits<T,D>::assign(val, m_img.m_data, m_img.channelstride());
            return val;
        }
        const T *operator&() const { return m_img.m_data; }
    };/*}}}*/

    template <int D>
    class const_pixel_proxy<D, typename enable_if<(D<=0 || D>=5)>::type>/*{{{*/
    {
    public:
        HOSTDEV const_pixel_proxy(const dimage_ptr &img) {}
    };/*}}}*/

    typedef typename copy_const<T,typename pixel_traits<T,1>::texel_type>::type
        texel_type;
    typedef typename remove_const<texel_type>::type mutable_texel_type;

    HOSTDEV dimage_ptr(texel_type *data, int width, int height, int rowstride);

    template <class P>
    HOSTDEV dimage_ptr(const dimage_ptr<P,C> &that,
               typename enable_if<is_convertible<P,T>::value>::type* =NULL);

    HOSTDEV int width() const { return m_width; }
    HOSTDEV int height() const { return m_height; }
    HOSTDEV int rowstride() const { return m_rowstride; }
    HOSTDEV int channelstride() const { return rowstride()*height(); }

    HOSTDEV bool empty() const { return rowstride()==0 || height()==0; }

    HOSTDEV int offset_at(int x, int y) const { return y*rowstride()+x; }
    HOSTDEV int offset_at(int2 pos) const { return offset_at(pos.x, pos.y); }

    HOSTDEV bool is_inside(int x, int y) const
        { return x < width() && y < height(); }
    HOSTDEV bool is_inside(int2 pos) const { return is_inside(pos.x, pos.y); }

    void copy_to_host(mutable_texel_type *out, int rowstride=0) const;

    void copy_to_host(std::vector<mutable_texel_type> &out, int rowtride=0) const;

    void copy_from_host(const texel_type *in, int w, int h, int rs=0);
    void copy_from_host(const std::vector<mutable_texel_type> &in,
                        int width, int height, int rowstride=0);

    dimage_ptr &operator=(dimage_ptr<const T,C> img);

    template <class U>
    typename enable_if<is_same<T,U>::value && !is_const<T>::value,dimage_ptr<U,C> &>::type
        operator=(const dimage_ptr<U,C> &img)
    {
        return operator=(static_cast<dimage_ptr<const T,C> >(img));
    }

    dimage_ptr &operator=(const dimage<T,C> &img) { return *this = &img; }

    HOSTDEV dimage_ptr<T,1> operator[](int i);

    HOSTDEV dimage_ptr<const T,1> operator[](int i) const;

    HOSTDEV operator texel_type*() { return m_data; }
    HOSTDEV operator const texel_type*() const { return m_data; }

    HOSTDEV texel_type *operator&() { return m_data; }
    HOSTDEV const texel_type *operator&() const { return m_data; }

    // These two here need to be inline to cope with VC++2008 non-conformance
    HOSTDEV pixel_proxy<C> operator*()
        { return pixel_proxy<C>(*this); }

    HOSTDEV const_pixel_proxy<C> operator*() const
        { return const_pixel_proxy<C>(*this); }

    HOSTDEV dimage_ptr &operator++();

    HOSTDEV dimage_ptr operator++(int);

    HOSTDEV dimage_ptr &operator--();

    HOSTDEV dimage_ptr operator--(int);

    HOSTDEV dimage_ptr &operator+=(int off);

    HOSTDEV dimage_ptr &operator-=(int off) { return operator+=(-off); }

private:
    template <class U, int D>
    friend class dimage_ptr;

    int m_width, m_height, m_rowstride;
    texel_type *m_data;
};

template <class T>
void copy_to_array(cudaArray *out, dimage_ptr<const T> in);

template <class T>
void copy_to_array(cudaArray *out, dimage_ptr<T> in)
{
    copy_to_array(out, dimage_ptr<const T>(in));
}

template <class T>
void copy_from_array(dimage_ptr<T> out, const cudaArray *in);

} // namespace rod

#include "dimage.hpp"


#endif
