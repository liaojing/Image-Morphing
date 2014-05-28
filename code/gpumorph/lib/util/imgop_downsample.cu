#include <cuda.h>
#include "filter_traits.h"
#include "image_ops.h"
#include "bspline3.h"
#include "mitchell_netravali.h"
#include "symbol.h"
#include "box_sampler.h"
#include "cubic_sampler.h"
#include "effects.h"

#define USE_LAUNCH_BOUNDS 1
const int BW_F1 = 32, // cuda block width
          BH_F1 = 8; 

const int BW_F2 = 32,
          BH_F2 = 8; 

#if USE_LAUNCH_BOUNDS
const int 
#if CUDA_SM >= 20
          NB_F1 = 2,  // number of blocks resident per SM
#else
          NB_F1 = 1,  // number of blocks resident per SM
#endif
          NB_F2 = 4;
#endif


const int KS = 4,
          MAX_SAMPDIM = 65536/(KS*KS*sizeof(float)); // gpu has 64k of cmem
__constant__ float c_kernel[MAX_SAMPDIM*KS*KS];

namespace rod
{

// downsample 

template <int C>
struct sum_traits
    : pixel_traits<float,C+1>
{
    // C weighed channels + total weight
    typedef typename pixel_traits<float,C+1>::pixel_type type;
};

template <class S, int C>
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F1*BH_F1, NB_F1)
#endif
__global__ 
void kernel_downsample1(dimage_ptr<typename sum_traits<C>::type,KS*KS> out,/*{{{*/
                        float2 invrate)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F1+tx, y = blockIdx.y*BH_F1+ty;

    if(!out.is_inside(x,y))
        return;

    // output will point to the pixel we're processing now
    int idx = out.offset_at(x,y);
    out += idx;

    // we're using some smem as registers not to blow up the register space,
    // here we define how much 'registers' are in smem, the rest is used
    // in regular registers
    
    typedef filter_traits<C> cfg;

    typedef typename sum_traits<C>::type sum_type;
    typedef typename pixel_traits<float,C>::pixel_type pixel_type;

    const int SMEM_SIZE = cfg::smem_size,
              REG_SIZE = KS*KS-SMEM_SIZE;

    __shared__ sum_type _sum[BH_F1][SMEM_SIZE][BW_F1];
    sum_type (*ssum)[BW_F1] = (sum_type (*)[BW_F1]) &_sum[ty][0][tx];

    sum_type sum[REG_SIZE];

    // Init registers to zero
    for(int i=0; i<REG_SIZE; ++i)
        sum[i] = sum_traits<C>::make_pixel(0);

#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *ssum[i] = sum_traits<C>::make_pixel(0);

    // top-left position of the kernel support
    float2 p = (make_float2(x,y)-1.5f)*invrate + 0.5;

    float *kernel = c_kernel;

    S sampler;

    int mx = ceil(invrate.x),
        my = ceil(invrate.y);

    for(int i=0; i<my; ++i)
    {
        for(int j=0; j<mx; ++j)
        {
            pixel_type value = sampler(p+make_float2(i,j));

            // scans through the kernel support, collecting data for each
            // position
#pragma unroll
            for(int i=0; i<SMEM_SIZE; ++i)
            {
                float wij = kernel[i];

                *ssum[i] += sum_traits<C>::make_pixel(value*wij, wij);
            }
            kernel += SMEM_SIZE;
#pragma unroll
            for(int i=0; i<REG_SIZE; ++i)
            {
                float wij = kernel[i];

                sum[i] += sum_traits<C>::make_pixel(value*wij, wij);
            }
            kernel += REG_SIZE;
        }
    }

    // writes out to gmem what's in the registers
#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *out[i] = *ssum[i];

#pragma unroll
    for(int i=0; i<REG_SIZE; ++i)
        *out[SMEM_SIZE+i] = sum[i];
}/*}}}*/

#ifdef _MSC_VER
	template <class T, int C, class U>
	#if USE_LAUNCH_BOUNDS
	__launch_bounds__(BW_F2*BH_F2, NB_F2)
	#endif
	__global__
	void kernel_downsample2(dimage_ptr<T,C> out, /*{{{*/
		                dimage_ptr<U,KS*KS> in)
#else
	template <class T, int C>
	#if USE_LAUNCH_BOUNDS
	__launch_bounds__(BW_F2*BH_F2, NB_F2)
	#endif
	__global__
	void kernel_downsample2(dimage_ptr<T,C> out, /*{{{*/
		                dimage_ptr<const typename sum_traits<pixel_traits<T,C>::components>::type,KS*KS> in)
#endif
{
    const int COMP = pixel_traits<T,C>::components;

    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F2+tx, y = blockIdx.y*BH_F2+ty;

    // out of bounds? goodbye
    if(!in.is_inside(x,y))
        return;

    // in and out points to the input/output pixel we're processing
    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    // treat corner cases where the support is outside the image
    int mi = min(y+KS,in.height())-y,
        mj = min(x+KS,in.width())-x;

    // sum the contribution of nearby pixels
    typename sum_traits<COMP>::type sum = sum_traits<COMP>::make_pixel(0);

#pragma unroll
    for(int i=0; i<mi; ++i)
    {
#pragma unroll
        for(int j=0; j<mj; ++j)
        {
            sum += *in[i*KS+j];
            ++in;
        }
        in += in.rowstride()-mj;
    }

    *out = filter_traits<COMP>::normalize_sum(sum);
}/*}}}*/

template <class T, int C>
void downsample(dimage_ptr<T,C> out, const cudaArray *in, /*{{{*/
                size_t w, size_t h, Interpolation interp)
{
    float2 rate = make_float2((float)out.width()/w,
                              (float)out.height()/h),
           invrate = 1.0f/rate;

    int2 ir = make_int2((int)ceil(invrate.x),(int)ceil(invrate.y));

    float (*kernfunc)(float v) = NULL;
    switch(interp)
    {
    case INTERP_BSPLINE3:
        kernfunc = bspline3;
        break;
    case INTERP_MITCHELL_NETRAVALI:
        kernfunc = mitchell_netravali;
        break;
    }

    assert(kernfunc != NULL);

    std::vector<float> kernel(ir.x*ir.y*KS*KS);

    for(int i=0; i<ir.y; ++i)
    {
        for(int j=0; j<ir.x; ++j)
        {
            for(int y = 0; y<KS; ++y)
            {
                for(int x = 0; x<KS; ++x)
                {
                    kernel[(i*ir.x + j)*KS*KS + y*KS+x]
                        = kernfunc(x+j*rate.x-1.5f)*
                          kernfunc(y+i*rate.y-1.5f)/(ir.x*ir.y);
                }
            }
        }
    }

    copy_to_symbol(c_kernel,kernel);

    typedef typename pixel_traits<T,C>::texel_type texel_type;
    typedef filter_traits<pixel_traits<T,C>::components> filter;

    filter::tex().normalized = false;
    filter::tex().addressMode[0] = filter::tex().addressMode[1] 
        = cudaAddressModeClamp;

    filter::tex().filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(filter::tex(), in);

    dim3 bdim(BW_F1,BH_F1),
         gdim((out.width()+bdim.x-1)/bdim.x, (out.height()+bdim.y-1)/bdim.y);

    typedef typename filter::texfetch_type texfetch;
    typedef typename sum_traits<pixel_traits<T,C>::components>::type sum_type;

    dimage<sum_type, KS*KS> temp(out.width(), out.height());

    kernel_downsample1<box_sampler<texfetch>, pixel_traits<T,C>::components>
        <<<gdim,bdim>>>(&temp, invrate);

    bdim = dim3(BW_F2,BH_F2);
    gdim = dim3((out.width()+bdim.x-1)/bdim.x, (out.height()+bdim.y-1)/bdim.y);

    kernel_downsample2<<<gdim, bdim>>>(out, &temp);

    cudaUnbindTexture(filter::tex());
}/*}}}*/

template <class T, int C>
void downsample(dimage_ptr<T,C> out, dimage_ptr<const T,C> in, /*{{{*/
              Interpolation interp)
{
    typedef typename pixel_traits<T,C>::texel_type texel_type;

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<texel_type>();

    cudaArray *a_in;

    cudaMallocArray(&a_in, &ccd, in.width(), in.height());

    copy_to_array(a_in, in);

    downsample(out, a_in, in.width(), in.height());

    cudaFreeArray(a_in);
}/*}}}*/

template
void downsample(dimage_ptr<float3,1> out, dimage_ptr<const float3,1> in, 
              Interpolation interp);

template
void downsample(dimage_ptr<float,1> out, dimage_ptr<const float,1> in, 
              Interpolation interp);

template
void downsample(dimage_ptr<float3,1> out, const cudaArray *in, 
                size_t w, size_t h, Interpolation interp);

template
void downsample(dimage_ptr<float,1> out, const cudaArray *in, 
                size_t w, size_t h, Interpolation interp);


} // namespace rod
