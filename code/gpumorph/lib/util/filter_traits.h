#ifndef NLFILTER_FILTER_TRAITS_H
#define NLFILTER_FILTER_TRAITS_H

#include <cuda.h>
#include "config.h"
#include "cuda_traits.h"
#include "dimage_fwd.h"

namespace rod
{

template <int C>
struct filter_traits {};

// gray images ------------------------------------------------------------

texture<float, 2, cudaReadModeElementType> t_in_gray;

struct texfetch_gray
{   
    typedef float result_type;

    __device__ float operator()(float x, float y)
    {
        return tex2D(t_in_gray, x, y);
    }
};

template <> 
struct filter_traits<1>
{
    typedef texfetch_gray texfetch_type;
    static const int smem_size = 3;

    static texture<float,2,cudaReadModeElementType> &tex() { return t_in_gray; }

    __device__ static float normalize_sum(float2 sum)
    {
        return sum.x / sum.y;
    }
};

// alpha_gray images ------------------------------------------------------------

texture<float2, 2, cudaReadModeElementType> t_in_alpha_gray;

struct texfetch_alpha_gray
{   
    typedef float2 result_type;

    __device__ float2 operator()(float x, float y)
    {
        return tex2D(t_in_alpha_gray, x, y);
    }
};

template <> 
struct filter_traits<2>
{
    typedef texfetch_alpha_gray texfetch_type;
    static const int smem_size = 3;

    static texture<float2,2,cudaReadModeElementType> &tex() 
        { return t_in_alpha_gray; }

    __device__ static float2 normalize_sum(float3 sum)
    {
        return make_float2(sum.x, sum.y) / sum.z;
    }
};

// rgb images ------------------------------------------------------------

texture<float4, 2, cudaReadModeElementType> t_in_rgb;

struct texfetch_rgb
{
    typedef float3 result_type;
    __device__ float3 operator()(float x, float y)
    {
        return make_float3(tex2D(t_in_rgb, x, y));
    }

};

template <>
struct filter_traits<3>
{
    typedef texfetch_rgb texfetch_type;
#if CUDA_SM >= 20
    static const int smem_size = 5;
#else
    static const int smem_size = 3;
#endif


    static texture<float4,2,cudaReadModeElementType> &tex() { return t_in_rgb;}

    __device__ static float3 normalize_sum(float4 sum)
    {
        return make_float3(sum) / sum.w;
    }
};


// rgba images ------------------------------------------------------------

texture<float4, 2, cudaReadModeElementType> t_in_rgba;

struct texfetch_rgba
{
    typedef float4 result_type;

    __device__ float4 operator()(float x, float y)
    {
        return tex2D(t_in_rgba, x, y);
    }
};

template <>
struct filter_traits<4>
{
    typedef texfetch_rgba texfetch_type;

    static texture<float4,2,cudaReadModeElementType> &tex() { return t_in_rgba;}
};

} // namespace rod

#endif
