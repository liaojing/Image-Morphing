#include <cuda.h>
#include "image_ops.h"
#include "filter_traits.h"
#include "bspline3.h"
#include "cubic_sampler.h"
#include "box_sampler.h"
#include "mitchell_netravali.h"

namespace rod
{

const int BW = 32, // cuda block width
          BH = 16; // cuda block height
          //NB = 3;

// upsample 
template <class S, class T, int C>
__global__ void kernel_upsample(dimage_ptr<T,C> out, float tw, float th)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!out.is_inside(x,y))
        return;

    S sampler;

    out += out.offset_at(x,y);

    *out = sampler(make_float2((x+0.5f)*tw, (y+0.5f)*th));
}


template <class T, int C>
void upsample(dimage_ptr<T,C> out, dimage_ptr<const T,C> in, 
              Interpolation interp)
{
    typedef typename pixel_traits<T,C>::texel_type texel_type;
    typedef filter_traits<pixel_traits<T,C>::components> filter;

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<texel_type>();

    cudaArray *a_in;

    cudaMallocArray(&a_in, &ccd, in.width(), in.height());

    copy_to_array(a_in, in);

    filter::tex().normalized = false;
    filter::tex().addressMode[0] = filter::tex().addressMode[1] 
        = cudaAddressModeClamp;

    switch(interp)
    { 
    case INTERP_BSPLINE3:
    case INTERP_MITCHELL_NETRAVALI:
    case INTERP_LINEAR:
        filter::tex().filterMode = cudaFilterModeLinear;
        break;
    case INTERP_BOX:
        filter::tex().filterMode = cudaFilterModePoint;
        break;
    }

    cudaBindTextureToArray(filter::tex(), a_in);

    dim3 bdim(BW,BH),
         gdim((out.width()+bdim.x-1)/bdim.x, (out.height()+bdim.y-1)/bdim.y);

    typedef typename filter::texfetch_type texfetch;

    float tw = (float)in.width()/out.width(),
          th = (float)in.height()/out.height();

    switch(interp)
    { 
    case INTERP_BSPLINE3:
        kernel_upsample<cubic_sampler<bspline3_weights, texfetch> > 
            <<<gdim,bdim>>>(out, tw, th);
        break;
    case INTERP_MITCHELL_NETRAVALI:
        kernel_upsample<cubic_sampler<mitchell_netravali_weights, texfetch> > 
            <<<gdim,bdim>>>(out, tw, th);
        break;
    case INTERP_LINEAR:
    case INTERP_BOX:
        kernel_upsample<box_sampler<texfetch> > 
            <<<gdim,bdim>>>(out, tw, th);
        break;
    }

    cudaUnbindTexture(filter::tex());

    cudaFreeArray(a_in);
}

template
void upsample(dimage_ptr<float4,1> out, dimage_ptr<const float4,1> in, 
              Interpolation interp);

template
void upsample(dimage_ptr<float2,1> out, dimage_ptr<const float2,1> in, 
              Interpolation interp);


}
