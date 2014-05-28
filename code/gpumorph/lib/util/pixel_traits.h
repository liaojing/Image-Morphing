#ifndef GPUFILTER_PIXEL_TRAITS_H
#define GPUFILTER_PIXEL_TRAITS_H

#include "cuda_traits.h"

namespace rod
{

template <class T, int D=1> 
struct pixel_traits;

namespace detail
{
    template <class T, int C>
    struct assign_helper;

    template <class T>
    struct assign_helper<T,1>
    {
        template <class U>
        HOSTDEV
        static void assign(T *pix, int stride, const U &data)
        {
            *pix = cuda_traits<T>::make(data);
        }
        template <class U>
        HOSTDEV
        static void assign(U &data, const T *pix, int stride)
        {
            data = cuda_traits<U>::make(*pix);
        }
    };

    template <class T>
    struct assign_helper<T,2>
    {
    private:
        typedef typename make_cuda_type<T,2>::type pixel_type_;
        typedef typename cuda_traits<T>::base_type base_type_;
    public:

        HOSTDEV
        static void assign(base_type_ *pix, int stride, const pixel_type_ &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
        }
        HOSTDEV
        static void assign(pixel_type_ &data, const base_type_ *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
        }
    };

    template <class T>
    struct assign_helper<T,3>
    {
    private:
        typedef typename make_cuda_type<T,3>::type pixel_type_;
        typedef typename cuda_traits<T>::base_type base_type_;
    public:

        HOSTDEV
        static void assign(base_type_ *pix, int stride, const pixel_type_ &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
            pix[stride*2] = data.z;
        }
        HOSTDEV
        static void assign(pixel_type_ &data, const base_type_ *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
            data.z = pix[stride*2];
        }
    };

    template <class T>
    struct assign_helper<T,4>
    {
    private:
        typedef typename make_cuda_type<T,4>::type pixel_type_;
        typedef typename cuda_traits<T>::base_type base_type_;
    public:

        HOSTDEV
        static void assign(base_type_ *pix, int stride, const pixel_type_ &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
            pix[stride*2] = data.z;
            pix[stride*3] = data.w;
        }
        HOSTDEV
        static void assign(pixel_type_ &data, const base_type_ *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
            data.z = pix[stride*2];
            data.w = pix[stride*3];
        }
    };

    // base for all pixel_traits
    template <class T, int C>
    struct pixtraits_common1
    {
        static const int planes = C;

        typedef typename make_cuda_type<T,C>::type pixel_type;
        typedef typename cuda_traits<pixel_type>::texel_type texel_type;
        typedef typename cuda_traits<T>::base_type base_type;
        
        static const int components = cuda_traits<pixel_type>::components;

        static const bool is_integral = rod::is_integral<base_type>::value;
    };

    // base for non composite types
    template <class T, int C, class EN=void>
    struct pixtraits_common2
        : pixtraits_common1<T,C>
        , assign_helper<T,C>
    {
    };

    // base for composite types
    template <class T>
    struct pixtraits_common2<T,1,
        typename enable_if<!is_const<T>::value && 
                           !is_volatile<T>::value && 
                           cuda_traits<T>::is_composite>::type>
        : pixel_traits<typename cuda_traits<T>::base_type,
                       cuda_traits<T>::components>
        , assign_helper<T,1>
    {
        using assign_helper<T,1>::assign;
    };

    template <class T, int C>
    struct pixtraits_common2<const T, C> : pixtraits_common2<T,C> {};

    template <class T, int C>
    struct pixtraits_common2<volatile T, C> : pixtraits_common2<T,C> {};
};

template <class T, int C> 
struct pixel_traits
    : detail::pixtraits_common2<T,C>
{
    typedef typename detail::pixtraits_common2<T,C>::pixel_type pixel_type;

    template <class X>
    HOSTDEV
    static pixel_type make_pixel(X x)
    {
        return cuda_traits<pixel_type>::make(x);
    }

    template <class X, class Y>
    HOSTDEV
    static pixel_type make_pixel(X x, Y y)
    {
        return cuda_traits<pixel_type>::make(x,y);
    }

    template <class X, class Y, class Z>
    HOSTDEV
    static pixel_type make_pixel(X x, Y y, Z z)
    {
        return cuda_traits<pixel_type>::make(x,y,z);
    }

    template <class X, class Y, class Z, class W>
    HOSTDEV
    static pixel_type make_pixel(X x, Y y, Z z, W w)
    {
        return cuda_traits<pixel_type>::make(x,y,z,w);
    }
};

} // namespace rod

#endif
