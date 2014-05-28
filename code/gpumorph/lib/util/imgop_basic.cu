#include <complex>
#include "symbol.h"
#include "dmath.h"
#include "effects.h"
#include "image_ops.h"

#define USE_LAUNCH_BOUNDS 1

namespace rod
{

const int BW = 32, // cuda block width
          BH = 16, // cuda block height
          NB = 3;

template <class XFORM, class TO, int C, class FROM, int D>
__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void kernel(dimage_ptr<TO,C> out, dimage_ptr<const FROM,D> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    XFORM xform;

    *out = xform(*in);
}

template <template<class,class> class XFORM, 
          class TO, int C, class FROM, int D>
void call_kernel(dimage_ptr<TO,C> out, dimage_ptr<const FROM,D> in)
{
    if(out.width() != in.width() || out.height() != in.height())
        throw std::runtime_error("Image dimensions don't match");

    dim3 bdim(BW,BH),
         gdim((in.width()+bdim.x-1)/bdim.x, (in.height()+bdim.y-1)/bdim.y);

    typedef XFORM<typename pixel_traits<TO,C>::pixel_type,
                  typename pixel_traits<FROM,D>::pixel_type> xform;

    kernel<xform><<<gdim, bdim>>>(out, in);
}

//{{{ conversion between pixel types --------------------------------------

template <class TO, class FROM>
struct convert_xform_norescale
{
    __device__ TO operator()(const FROM &p) const
    {
        return pixel_traits<TO>::make_pixel(p);
    }
};

template <class TO, class FROM, class EN=void>
struct convert_xform : convert_xform_norescale<TO,FROM>
{
};

template <class TO, class FROM>
struct convert_xform<TO,FROM,
    typename enable_if<pixel_traits<FROM>::is_integral && 
                      !pixel_traits<TO>::is_integral>::type>
{
    __device__ TO operator()(const FROM &p) const
    {
        return pixel_traits<TO>::make_pixel(p)/255.0f;
    }
};

template <class TO, class FROM>
struct convert_xform<TO,FROM,
    typename enable_if<!pixel_traits<FROM>::is_integral && 
                       pixel_traits<TO>::is_integral>::type>
{
    __device__ TO operator()(const FROM &p) const
    {
        return pixel_traits<TO>::make_pixel(saturate(p)*255.0f+0.5f);
    }
};

template <class TO, class FROM>
struct convert_xform2 : convert_xform<TO,FROM> {};

template <class TO, int C, class FROM, int D>
void convert(dimage_ptr<TO,C> out, dimage_ptr<const FROM,D> in, bool rescale)
{
    if(rescale)
        call_kernel<convert_xform2>(out, in);
    else
        call_kernel<convert_xform_norescale>(out, in);
}

template void convert(dimage_ptr<float3> out, dimage_ptr<const float,3> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const float,3> in, bool rescale);

template void convert(dimage_ptr<float3> out, dimage_ptr<const uchar3> in, bool rescale);

template void convert(dimage_ptr<float4> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<float4> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<float4> out, dimage_ptr<const uchar3> in, bool rescale);
template void convert(dimage_ptr<float4> out, dimage_ptr<const uchar4> in, bool rescale);
template void convert(dimage_ptr<float4> out, dimage_ptr<const float3> in, bool rescale);

template void convert(dimage_ptr<float3> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<float3> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<float3> out, dimage_ptr<const uchar4> in, bool rescale);

template void convert(dimage_ptr<uchar4> out, dimage_ptr<const float4> in, bool rescale);

template void convert(dimage_ptr<float,3> out, dimage_ptr<const float3> in, bool rescale);
template void convert(dimage_ptr<float,3> out, dimage_ptr<const uchar3> in, bool rescale);

template void convert(dimage_ptr<float3> out, dimage_ptr<const float> in, bool rescale);
template void convert(dimage_ptr<float,3> out, dimage_ptr<const float> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const float> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const float3> in, bool rescale);
template void convert(dimage_ptr<unsigned char> out, dimage_ptr<const float> in, bool rescale);
template void convert(dimage_ptr<float3> out, dimage_ptr<const float4> in, bool rescale);
template void convert(dimage_ptr<float1> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<float1> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<float1> out, dimage_ptr<const uchar4> in, bool rescale);
template void convert(dimage_ptr<float> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<float> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<float> out, dimage_ptr<const uchar3> in, bool rescale);
template void convert(dimage_ptr<float> out, dimage_ptr<const uchar4> in, bool rescale);
template void convert(dimage_ptr<unsigned char> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<unsigned char> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<unsigned char> out, dimage_ptr<const uchar3> in, bool rescale);
template void convert(dimage_ptr<unsigned char> out, dimage_ptr<const uchar4> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const unsigned char> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const uchar1> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const uchar2> in, bool rescale);
template void convert(dimage_ptr<uchar3> out, dimage_ptr<const uchar4> in, bool rescale);
template void convert(dimage_ptr<float> out, dimage_ptr<const unsigned char> in, bool rescale);
/*}}}*/

//{{{ lrgb2srgb ------------------------------------------------------------

template <class TO, class FROM>
struct lrgb2srgb_xform
{
    __device__ TO operator()(const FROM &p) const
    {
        return lrgb2srgb(p);
    }
};

template <class TO, int C, class FROM, int D>
void lrgb2srgb(dimage_ptr<TO,C> out, dimage_ptr<const FROM,D> in)
{
    call_kernel<lrgb2srgb_xform>(out, in);
}

template void lrgb2srgb(dimage_ptr<float3> out, dimage_ptr<const float3> in);
template void lrgb2srgb(dimage_ptr<float3> out, dimage_ptr<const float,3> in);
template void lrgb2srgb(dimage_ptr<float,3> out, dimage_ptr<const float,3> in);
template void lrgb2srgb(dimage_ptr<float> out, dimage_ptr<const float> in);
/*}}}*/

//{{{ grayscale ------------------------------------------------------------

template <class FROM, class EN=void>
struct grayscale_xform_base
{
    __device__ float operator()(const FROM &p) const
    {
        return grayscale(p);
    }
};

template <class FROM>
struct grayscale_xform_base<FROM,
    typename enable_if<pixel_traits<FROM>::is_integral>::type>
{
    __device__ float operator()(const FROM &p) const
    {
        return grayscale(make_float3(p)/255.0f);
    }
};

template <class TO, class FROM>
struct grayscale_xform;

template <class FROM>
struct grayscale_xform<float,FROM> : grayscale_xform_base<FROM> {};

template <class FROM, int D>
void grayscale(dimage_ptr<float> out, dimage_ptr<const FROM,D> in)
{
    call_kernel<grayscale_xform>(out, in);
}

template void grayscale(dimage_ptr<float> out, dimage_ptr<const uchar3> in);
template void grayscale(dimage_ptr<float> out, dimage_ptr<const float3> in);
template void grayscale(dimage_ptr<float> out, dimage_ptr<const float,3> in);
/*}}}*/

//{{{ luminance ------------------------------------------------------------

template <class TO, class FROM> struct luminance_xform;

template <class FROM>
struct luminance_xform<float,FROM>
{
    __device__ float operator()(const FROM &p) const
    {
        return luminance(p);
    }
};

template <class FROM, int D>
void luminance(dimage_ptr<float> out, dimage_ptr<const FROM,D> in)
{
    call_kernel<luminance_xform>(out, in);
}

template void luminance(dimage_ptr<float> out, dimage_ptr<const float3> in);
template void luminance(dimage_ptr<float> out, dimage_ptr<const float,3> in);
template<> void luminance(dimage_ptr<float> out, dimage_ptr<const float> in)
{
    out = in;
}

/*}}}*/

} // namespace rod
