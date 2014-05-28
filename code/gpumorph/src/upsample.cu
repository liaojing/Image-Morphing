#include <cuda.h>
#include <util/dimage.h>
#include <util/timer.h>
#include <util/image_ops.h>
#include "pyramid.h"

__global__ void conv_to_block_of_arrays(float2 *v,
                                        rod::dimage_ptr<const float2> in,
                                        KernPyramidLevel lvl,
                                        float2 m)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!in.is_inside(pos))
        return;

    in += in.offset_at(pos);
    v += mem_index(lvl, pos);

    *v = *in * m;
}

void upsample(PyramidLevel &dest, PyramidLevel &orig)
{
    rod::base_timer &timer
        = rod::timers.gpu_add("upsample",dest.width*dest.height,"P");

    rod::dimage<float2> vec_orig;
    internal_vector_to_image(vec_orig, orig.v, orig, 
                             make_float2(1,1));

    rod::dimage<float2> vec_dest(dest.width, dest.height);

    rod::upsample(&vec_dest, &vec_orig, rod::INTERP_LINEAR);

    dest.v.fill(0);

    dim3 bdim(32,8),
         gdim((dest.width+bdim.x-1)/bdim.x,
                (dest.height+bdim.y-1)/bdim.y);

    conv_to_block_of_arrays<<<gdim, bdim>>>(&dest.v, &vec_dest, dest,
                  make_float2((float)dest.width/orig.width,
                              (float)dest.height/orig.height));

    timer.stop();
}
