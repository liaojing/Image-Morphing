#include "symbol.h"
#include "dmath.h"
#include "image_ops.h"
#include "config.h"

namespace rod
{

const int BW = 32, // cuda block width
          BH = 16, // cuda block height
          NB = 3;

__constant__ float c_conv_kernel[20]; // max kernel diameter == 20
texture<float, 2, cudaReadModeElementType> t_in_convolution_float;
texture<float4, 2, cudaReadModeElementType> t_in_convolution_float4;

template <class T>
struct texture_traits { };

template <>
struct texture_traits<float>
{
    static texture<float, 2, cudaReadModeElementType> &
        get() { return t_in_convolution_float; }

    __device__ static float tex2D(float x, float y)
    {
        return ::tex2D(t_in_convolution_float, x, y);
    }
};

template <>
struct texture_traits<float4>
{
    static texture<float4, 2, cudaReadModeElementType> &
        get() { return t_in_convolution_float4; }

    __device__ static float4 tex2D(float x, float y)
    {
        return ::tex2D(t_in_convolution_float4, x, y);
    }
};

template<int R, class T, class U>
__device__
void load_convolve_rows(T *s_in, int tx, U tu, U tv) /*{{{*/
{
    typedef pixel_traits<T> pix_traits;
    typedef texture_traits<typename pix_traits::texel_type> tex;

    // load middle data
    s_in[R + tx] = pix_traits::make_pixel(tex::tex2D(tu, tv));

    // load left and right data
    if(R <= BW/2) 
    {
        if(tx < R) 
            s_in[tx] = pix_traits::make_pixel(tex::tex2D(tu - R, tv));
        else if(tx < R*2) 
            s_in[BW+tx] = pix_traits::make_pixel(tex::tex2D(tu - R+BW, tv));
    } 
    else if(R <= BW) 
    {
        if(tx < R) 
        {
            s_in[tx] = pix_traits::make_pixel(tex::tex2D(tu - R, tv));
            s_in[R+BW + tx] = pix_traits::make_pixel(tex::tex2D(tu + BW, tv));
        }
    } 
    else 
    {
#pragma unroll
        for (int i = 0; i < (R+BW-1)/BW; ++i) 
        {
            int wx = i*BW+tx;
            if( wx < R) 
            {
                s_in[wx] = pix_traits::make_pixel(tex::tex2D(tu - R + i*BW, tv));
                s_in[R+BW + wx] = pix_traits::make_pixel(tex::tex2D(tu + BW + i*BW, tv));
            }
        }
    }

    // convolve row
    T s = pix_traits::make_pixel(0.f);
    for (int k = -R; k <= R; ++k) 
        s += s_in[R + tx + k] * c_conv_kernel[k + R];

    s_in[R + tx] = s;
}/*}}}*/

template<int R,class T,int C>
__global__ __launch_bounds__(BW*BH, NB)
void convolution_kernel(dimage_ptr<T,C> out, float inv_norm,/*{{{*/
                        int scale)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    typedef typename pixel_traits<T,C>::pixel_type pixel_type;
    typedef pixel_traits<pixel_type> pix_traits;
    typedef texture_traits<typename pix_traits::texel_type> tex;


    float tu = x + .5f, tv = y + .5f;
    __shared__ pixel_type s_inblock[BH + R*2][BW + R*2];

    // load middle data
    load_convolve_rows<R>( &s_inblock[R + ty][0], tx, tu, tv);

    // load upper and lower data
    if(R <= BH/2) 
    {
        if(ty < R) 
            load_convolve_rows<R>(&s_inblock[ty][0], tx, tu, tv - R);
        else if(ty < R*2) 
            load_convolve_rows<R>(&s_inblock[BH + ty][0], tx, tu, tv - R + BH);
    } 
    else if(R <= BH) 
    {
        if(ty < R) 
        {
            load_convolve_rows<R>(&s_inblock[ty][0], tx, tu, tv - R);
            load_convolve_rows<R>(&s_inblock[R + BH + ty][0], tx, tu, tv + BH);
        }
    } 
    else 
    {
        for (int i = 0; i < (R+BH-1)/BH; ++i) 
        {
            int wy = i*BH+ty;
            if( wy < R ) 
            {
                load_convolve_rows<R>(&s_inblock[wy][0], tx, tu, tv - R + i*BH);
                load_convolve_rows<R>(&s_inblock[R + BH + wy][0], tx, tu, tv + BH + i*BH);
            }
        }
    }

    __syncthreads();

    tx *= scale;
    ty *= scale;

    if(tx >= BW || ty >= BH)
        return;

    x = (blockIdx.x*BW)/scale + threadIdx.x;
    y = (blockIdx.y*BH)/scale + threadIdx.y;

    if(!out.is_inside(x,y))
        return;

    out += out.offset_at(x,y);

    // convolve cols
    pixel_type s = pixel_traits<T,C>::make_pixel(0.f);
#pragma unroll
    for (int k = -R; k <= R; ++k)
        s += s_inblock[R + ty + k][R + tx] * c_conv_kernel[k + R];
    *out = s*inv_norm;
}/*}}}*/

template <class T, int C, class U, int D, int R>
void convolve(dimage_ptr<T,C> out, dimage_ptr<const U,D> in,/*{{{*/
              const array<float,R> &kernel, int scale)
{
    copy_to_symbol(c_conv_kernel,kernel);

    typedef typename pixel_traits<U>::texel_type texel_type;
    typedef texture_traits<texel_type> tex;

    cudaArray *a_in;
    cudaChannelFormatDesc ccd 
        = cudaCreateChannelDesc<texel_type>();

    cudaMallocArray(&a_in, &ccd, in.width(), in.height());

    tex::get().normalized = false;
    tex::get().filterMode = cudaFilterModePoint;

    tex::get().addressMode[0] = tex::get().addressMode[1] 
        = cudaAddressModeMirror;

    dim3 bdim(BW,BH),
         gdim((in.width()+bdim.x-1)/bdim.x, (in.height()+bdim.y-1)/bdim.y);

    float norm=0;
    for(int i=0; i<kernel.size(); ++i)
        norm += kernel[i];

    cudaBindTextureToArray(tex::get(), a_in);

    for(int c=0; c<D; ++c)

    {
        cudaMemcpy2DToArray(a_in, 0, 0, in[c], 
                            in.rowstride()*sizeof(texel_type),
                            in.width()*sizeof(texel_type), in.height(),
                            cudaMemcpyDeviceToDevice);
        if(D==1)
            convolution_kernel<R><<<gdim, bdim>>>(out,1/(norm*norm), scale);
        else
            convolution_kernel<R><<<gdim, bdim>>>(out[c],1/(norm*norm), scale);
    }
    cudaUnbindTexture(tex::get());
    cudaFreeArray(a_in);
}/*}}}*/

template void convolve(dimage_ptr<float,1> out, dimage_ptr<const float,1> in, 
                       const array<float,8> &kernel, int);

// lower sm doesn't have enough shared memory for RGB convolution
#if USE_SM>=20

template void convolve(dimage_ptr<float,3> out, dimage_ptr<const float3,1> in, 
                       const array<float,8> &kernel, int);

template void convolve(dimage_ptr<float3,1> out, dimage_ptr<const float3,1> in, 
                       const array<float,8> &kernel, int);


template void convolve(dimage_ptr<float,3> out, dimage_ptr<const float,3> in, 
                       const array<float,8> &kernel, int);
#endif

} // namespace rod
