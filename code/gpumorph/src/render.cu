#include <cuda.h>
#include "pyramid.h"
#include <util/dimage.h>
#include <util/timer.h>
#include <util/dmath.h>

texture<float4, 2, cudaReadModeElementType> tex_img0, tex_img1;

__global__ void kernel_render_halfway_image(rod::dimage_ptr<float3> out,
                                            KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;

    int idx = mem_index(lvl, pos);
    float2 v = lvl.v[idx];

    int d = 0;

    float2 p0 = pos-v;
    float4 c0;
    if(lvl.contains(p0))
    {
        c0 = tex2D(tex_img0, p0.x+0.5f, p0.y+0.5f);
        ++d;
    }
    else
        c0 = make_float4(0,0,0,0);

    float2 p1 = pos+v;
    float4 c1;
    if(lvl.contains(p1))
    {
        c1 = tex2D(tex_img1, p1.x+0.5f, p1.y+0.5f);
        ++d;
    }
    else
        c1 = make_float4(0,0,0,0);

    out += out.offset_at(pos);

    if(d == 2)
        *out = (c0+c1)/2;
    else if(d == 0)
        *out = make_float3(255,255,255);
    else
        *out = c0+c1;
}

void render_halfway_image(rod::dimage<float3> &out, PyramidLevel &lvl,
                          const cudaArray *img0, const cudaArray *img1)
{
    tex_img0.normalized = false;
    tex_img0.filterMode = cudaFilterModeLinear;
    tex_img0.addressMode[0] = tex_img0.addressMode[1] = cudaAddressModeClamp;

    tex_img1.normalized = false;
    tex_img1.filterMode = cudaFilterModeLinear;
    tex_img1.addressMode[0] = tex_img1.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex_img0, img0);
    cudaBindTextureToArray(tex_img1, img1);

    out.resize(lvl.width, lvl.height);

    dim3 bdim(32,8),
         gdim((lvl.width+bdim.x-1)/bdim.x,
              (lvl.height+bdim.y-1)/bdim.y);

    KernPyramidLevel klvl(lvl);

    rod::base_timer &timer 
        = rod::timers.gpu_add("render halfway",lvl.width*lvl.height,"P");

    kernel_render_halfway_image<<<gdim, bdim>>>(&out, klvl);

    timer.stop();

}

void render_halfway_image(rod::dimage<float3> &out, PyramidLevel &lvl,
                          const rod::dimage<float3> &in0,
                          const rod::dimage<float3> &in1)
{
    cudaArray *a_in0=NULL, *a_in1 = NULL;

    try
    {
        cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&a_in0, &ccd, in0.width(), in0.height());
        cudaMallocArray(&a_in1, &ccd, in1.width(), in1.height());

        copy_to_array(a_in0, &in0);
        copy_to_array(a_in1, &in1);

        render_halfway_image(out, lvl, a_in0, a_in1);

        cudaFreeArray(a_in0);
        cudaFreeArray(a_in1);
    }
    catch(...)
    {
        if(a_in0)
            cudaFreeArray(a_in0);
        if(a_in1)
            cudaFreeArray(a_in1);
    }
}

__global__ void kernel_render_halfway_image(rod::dimage_ptr<float3> out,
                                            rod::dimage_ptr<const float2> hwpar)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(pos.x >= out.width() || pos.y >= out.height())
        return;

    hwpar += hwpar.offset_at(pos);
    float2 v = *hwpar;

    int d = 0;

    float2 p0 = pos-v;
    float4 c0;
    if(p0.x >= 0 && p0.x < out.width() &&
       p0.y >= 0 && p0.y < out.height())
    {
        c0 = tex2D(tex_img0, p0.x+0.5f, p0.y+0.5f);
        ++d;
    }
    else
        c0 = make_float4(0,0,0,0);

    float2 p1 = pos+v;
    float4 c1;
    if(p1.x >= 0 && p1.x < out.width() &&
       p1.y >= 0 && p1.y < out.height())
    {
        c1 = tex2D(tex_img1, p1.x+0.5f, p1.y+0.5f);
        ++d;
    }
    else
        c1 = make_float4(0,0,0,0);

    out += out.offset_at(pos);

    if(d == 2)
        *out = (c0+c1)/2;
    else if(d == 0)
        *out = make_float3(255,255,255);
    else
        *out = c0+c1;
}

void render_halfway_image(rod::dimage<float3> &out,
                          const rod::dimage<float2> &hwpar,
                          const cudaArray *img0, const cudaArray *img1,
                          size_t width, size_t height)
{
    tex_img0.normalized = false;
    tex_img0.filterMode = cudaFilterModeLinear;
    tex_img0.addressMode[0] = tex_img0.addressMode[1] = cudaAddressModeClamp;

    tex_img1.normalized = false;
    tex_img1.filterMode = cudaFilterModeLinear;
    tex_img1.addressMode[0] = tex_img1.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex_img0, img0);
    cudaBindTextureToArray(tex_img1, img1);

    out.resize(width, height);

    dim3 bdim(32,8),
         gdim((width+bdim.x-1)/bdim.x,
              (height+bdim.y-1)/bdim.y);

    kernel_render_halfway_image<<<gdim, bdim>>>(&out, &hwpar);
}

void render_halfway_image(rod::dimage<float3> &out,
                          const rod::dimage<float2> &hwpar,
                          const rod::dimage<float3> &in0,
                          const rod::dimage<float3> &in1)
{
    cudaArray *a_in0=NULL, *a_in1 = NULL;

    try
    {
        cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&a_in0, &ccd, in0.width(), in0.height());
        cudaMallocArray(&a_in1, &ccd, in1.width(), in1.height());

        copy_to_array(a_in0, &in0);
        copy_to_array(a_in1, &in1);

        render_halfway_image(out, hwpar, a_in0, a_in1, in0.width(), in0.height());

        cudaFreeArray(a_in0);
        cudaFreeArray(a_in1);
    }
    catch(...)
    {
        if(a_in0)
            cudaFreeArray(a_in0);
        if(a_in1)
            cudaFreeArray(a_in1);
    }
}
