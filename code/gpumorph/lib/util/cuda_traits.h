#ifndef GPUFILTER_CUDA_TRAITS_H
#define GPUFILTER_CUDA_TRAITS_H

#include "dmath.h"
#include "hostdev.h"
#include "util.h"

namespace rod
{

template <class T>
struct cuda_traits
{
    typedef T base_type;
    typedef T texel_type;
    static const int components = 1;
    static const bool is_composite = false;

    HOSTDEV static T make(base_type x)
    {
        return x;
    }

    template <class U>
    HOSTDEV
    static
    typename enable_if<cuda_traits<U>::is_composite, T>::type
    make(const U &v)
    {
        return v.x;
    }
};

template <>
struct cuda_traits<uchar1>
{
    typedef unsigned char base_type;
    typedef unsigned char texel_type;
    static const int components = 1;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static uchar1 make(U x) { return make_uchar1(x); }
};

template <>
struct cuda_traits<uchar2>
{
    typedef unsigned char base_type;
    typedef uchar2 texel_type;
    static const int components = 2;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static uchar2 make(U x) { return make_uchar2(x); }

    template <class U, class V>
    HOSTDEV static uchar2 make(U x,V y) { return make_uchar2(x,y); }
};

template <>
struct cuda_traits<uchar3>
{
    typedef unsigned char base_type;
    typedef uchar4 texel_type;
    static const int components = 3;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static uchar3 make(U x) { return make_uchar3(x); }

    template <class U, class V>
    HOSTDEV static uchar3 make(U x,V y) { return make_uchar3(x,y); }

    template <class U, class V, class W>
    HOSTDEV static uchar3 make(U x,V y,W z) { return make_uchar3(x,y,z); }
};

template <>
struct cuda_traits<uchar4>
{
    typedef unsigned char base_type;
    typedef uchar4 texel_type;
    static const int components = 4;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static uchar4 make(U x) { return make_uchar4(x); }

    template <class U, class V>
    HOSTDEV static uchar4 make(U x,V y) { return make_uchar4(x,y); }

    template <class U, class V, class W>
    HOSTDEV static uchar4 make(U x,V y,W z) { return make_uchar4(x,y,z); }

    template <class U, class V, class W, class X>
    HOSTDEV static uchar4 make(U x,V y,W z,X w) { return make_uchar4(x,y,z,w); }
};

template <>
struct cuda_traits<float1>
{
    typedef float base_type;
    typedef float texel_type;
    static const int components = 1;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static float1 make(U x) { return make_float1(x); }
};

template <>
struct cuda_traits<float2>
{
    typedef float base_type;
    typedef float2 texel_type;
    static const int components = 2;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static float2 make(U x) { return make_float2(x); }

    template <class U, class V>
    HOSTDEV static float2 make(U x,V y) { return make_float2(x,y); }
};

template <>
struct cuda_traits<float3>
{
    typedef float base_type;
    typedef float4 texel_type;
    static const int components = 3;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static float3 make(U x) { return make_float3(x); }

    template <class U, class V>
    HOSTDEV static float3 make(U x,V y) { return make_float3(x,y); }

    template <class U, class V, class W>
    HOSTDEV static float3 make(U x,V y,W z) { return make_float3(x,y,z); }
};

template <>
struct cuda_traits<float4>
{
    typedef float base_type;
    typedef float4 texel_type;
    static const int components = 4;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static float4 make(U x) { return make_float4(x); }

    template <class U, class V>
    HOSTDEV static float4 make(U x,V y) { return make_float4(x,y); }

    template <class U, class V, class W>
    HOSTDEV static float4 make(U x,V y,W z) { return make_float4(x,y,z); }

    template <class U, class V, class W, class X>
    HOSTDEV static float4 make(U x,V y,W z,X w) { return make_float4(x,y,z,w); }
};

template <>
struct cuda_traits<double1>
{
    typedef double base_type;
    typedef float texel_type;
    static const int components = 1;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static double1 make(U x) { return make_double1(x); }
};

template <>
struct cuda_traits<double2>
{
    typedef double base_type;
    typedef float2 texel_type;
    static const int components = 2;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static double2 make(U x) { return make_double2(x); }

    template <class U, class V>
    HOSTDEV static double2 make(U x,V y) { return make_double2(x,y); }
};

template <>
struct cuda_traits<double3>
{
    typedef double base_type;
    typedef float4 texel_type;
    static const int components = 3;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static double3 make(U x) { return make_double3(x); }

    template <class U, class V>
    HOSTDEV static double3 make(U x,V y) { return make_double3(x,y); }

    template <class U, class V, class W>
    HOSTDEV static double3 make(U x,V y,W z) { return make_double3(x,y,z); }
};

template <>
struct cuda_traits<double4>
{
    typedef double base_type;
    typedef float4 texel_type;
    static const int components = 4;
    static const bool is_composite = true;

    template <class U>
    HOSTDEV static double4 make(U x) { return make_double4(x); }

    template <class U, class V>
    HOSTDEV static double4 make(U x,V y) { return make_double4(x,y); }

    template <class U, class V, class W>
    HOSTDEV static double4 make(U x,V y,W z) { return make_double4(x,y,z); }

    template <class U, class V, class W, class X>
    HOSTDEV static double4 make(U x,V y,W z,X w) { return make_double4(x,y,z,w); }
};

template <class T>
struct cuda_traits<const T> : cuda_traits<T> {};

template <class T>
struct cuda_traits<const volatile T> : cuda_traits<T> {};


/*}}}*/

template <class T, int C>
struct make_cuda_type
{
    typedef typename make_cuda_type<typename cuda_traits<T>::base_type, 
            cuda_traits<T>::components*C>::type type;
};

template <class T, int C>
struct make_cuda_type<const T, C>
{
    typedef const typename make_cuda_type<T,C>::type type;
};

template <class T, int C>
struct make_cuda_type<volatile T, C>
{
    typedef volatile typename make_cuda_type<T,C>::type type;
};

template <> struct make_cuda_type<float,1> { typedef float type; };
template <> struct make_cuda_type<float,2> { typedef float2 type; };
template <> struct make_cuda_type<float,3> { typedef float3 type; };
template <> struct make_cuda_type<float,4> { typedef float4 type; };

template <> struct make_cuda_type<double,1> { typedef double type; };
template <> struct make_cuda_type<double,2> { typedef double2 type; };
template <> struct make_cuda_type<double,3> { typedef double3 type; };
template <> struct make_cuda_type<double,4> { typedef double4 type; };

template <> struct make_cuda_type<unsigned char,1> { typedef unsigned char type; };
template <> struct make_cuda_type<unsigned char,2> { typedef uchar2 type; };
template <> struct make_cuda_type<unsigned char,3> { typedef uchar3 type; };
template <> struct make_cuda_type<unsigned char,4> { typedef uchar4 type; };

}

#endif
