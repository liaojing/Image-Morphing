#include <cmath>
#include <util/timer.h>
#include <util/dimage.h>
#include <util/image_ops.h>
#include <iostream>
#include <sstream>
#include <cstdio>  // for sprintf
#include <cerrno>  // for errno
#include <cstring> // for strerror
#include "morph.h"
#include "imgio.h"
#include "pyramid.h"

#define USE_IMAGEMAGICK 1

Pyramid::~Pyramid()
{
    for(size_t i=0; i<m_data.size(); ++i)
        delete m_data[i];
}
PyramidLevel &Pyramid::append_new(int w, int h)
{
    m_data.push_back(new PyramidLevel(w,h));
    return *m_data.back();
}

PyramidLevel::PyramidLevel(int w, int h)
    : width(w), height(h)
{
    // align on 128 byte boundary
    rowstride = (w + 31)/32 * 32;
    inv_wh = 1.0f/(w*h);

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&img0, &ccd, w, h);
    cudaMallocArray(&img1, &ccd, w, h);

    size_t size = rowstride*h;

    v.resize(size);

    ssim.cross.resize(size);
    ssim.luma.resize(size);
    ssim.mean.resize(size);
    ssim.var.resize(size);
    ssim.value.resize(size);
    ssim.counter.resize(size);

    tps.axy.resize(size);
    tps.b.resize(size);

    ui.axy.resize(size);
    ui.b.resize(size);

    impmask_rowstride = (w+4)/5+2;

    improving_mask.resize(impmask_rowstride*((h+4)/5+2));
}
PyramidLevel::~PyramidLevel()
{
    cudaFreeArray(img0);
    cudaFreeArray(img1);
}

KernPyramidLevel::KernPyramidLevel(PyramidLevel &lvl)
{
    ssim.cross = &lvl.ssim.cross;
    ssim.var = &lvl.ssim.var;
    ssim.mean = &lvl.ssim.mean;
    ssim.luma = &lvl.ssim.luma;
    ssim.value = &lvl.ssim.value;
    ssim.counter = &lvl.ssim.counter;

    tps.axy = &lvl.tps.axy;
    tps.b = &lvl.tps.b;

    ui.axy = &lvl.ui.axy;
    ui.b = &lvl.ui.b;

    v = &lvl.v;

    improving_mask = &lvl.improving_mask;

    rowstride = lvl.rowstride;
    impmask_rowstride = lvl.impmask_rowstride;

    pixdim = make_int2(lvl.width, lvl.height);
    inv_wh = lvl.inv_wh;
}


template <class T> 
T log2(const T &v)/*{{{*/
{
    using std::log;
    return log(v)/log(T(2));
}/*}}}*/


// VC < 10
#if defined(_MSC_VER) && _MSC_VER < 1600
inline float round(float v)
{
	return (float)(int)(v+0.5);
}
#endif


void create_pyramid(Pyramid &pyr,
                    const rod::dimage<float3> &img0,
                    const rod::dimage<float3> &img1,
                    int start_res, bool verbose)
{
    rod::base_timer &timer_total = rod::timers.gpu_add("Pyramid creation");

    size_t nlevels 
		= (size_t)(log2((float)std::min(img0.width(),img0.height())) - log2((float)start_res))+1;

    rod::base_timer *timer 
        = &rod::timers.gpu_add("level 0",img0.width()*img0.height(),"P");

    rod::dimage<float> luma0(img0.width(),img0.height()), 
                       luma1(img1.width(),img1.height());

    PyramidLevel &lvl0 = pyr.append_new(img0.width(), img0.height());

    luminance(&luma0, &img0);
    luminance(&luma1, &img1);

    copy_to_array(lvl0.img0, &luma0);
    copy_to_array(lvl0.img1, &luma1);

    timer->stop();

    for(size_t l=1; l<nlevels; ++l)
    {
        int w = (int)round(img0.width()/((float)(1<<l))),
            h = (int)round(img0.height()/((float)(1<<l)));

        if(verbose)
            std::clog << "Level " << l << ": " << w << "x" << h << std::endl;

        std::ostringstream ss;
        ss << "level " << l;

        rod::scoped_timer_stop sts(rod::timers.gpu_add(ss.str(), w*h,"P"));

        PyramidLevel &lvl = pyr.append_new(w,h);

        rod::dimage<float> luma(w,h);
        ::downsample(luma, luma0);
        copy_to_array(lvl.img0, &luma);

        ::downsample(luma, luma1);
        copy_to_array(lvl.img1, &luma);
    }

    timer_total.stop();
}

template <class T>
__global__ void internal_vector_to_image(rod::dimage_ptr<T> res,
                             const T *v, KernPyramidLevel lvl,
                             T mult)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!res.is_inside(pos))
        return;

    res += res.offset_at(pos);
    lvl.v += mem_index(lvl, pos);

    *res = *lvl.v * mult;
}


template <class T>
void internal_vector_to_image(rod::dimage<T> &dest,
                              const rod::dvector<T> &orig,
                              const PyramidLevel &lvl,
                              T mult)
{
    dim3 bdim(32,8),
         gdim((lvl.width+bdim.x-1)/bdim.x,
              (lvl.height+bdim.y-1)/bdim.y);

    dest.resize(lvl.width,lvl.height);

    // const cast is harmless
    KernPyramidLevel klvl(const_cast<PyramidLevel&>(lvl));

    dest.resize(lvl.width, lvl.height);

    internal_vector_to_image<<<gdim, bdim>>>(&dest, &orig, klvl, mult);
}

template
void internal_vector_to_image(rod::dimage<float> &dest,
                              const rod::dvector<float> &orig,
                              const PyramidLevel &lvl,
                              float mult);

template
void internal_vector_to_image(rod::dimage<float2> &dest,
                              const rod::dvector<float2> &orig,
                              const PyramidLevel &lvl,
                              float2 mult);

template <class T>
__global__ void image_to_internal_vector(T *v,
                                         rod::dimage_ptr<const T> in,
                                         KernPyramidLevel lvl,
                                         T mult)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!in.is_inside(pos))
        return;

    in += in.offset_at(pos);
    v += mem_index(lvl, pos);

    *v = *in * mult;
}

template <class T>
void image_to_internal_vector(rod::dvector<T> &dest,
                              const rod::dimage<T> &orig,
                              const PyramidLevel &lvl,
                              T mult)
{
    assert(lvl.width == orig.width());
    assert(lvl.height == orig.height());

    // const cast is harmless
    KernPyramidLevel klvl(const_cast<PyramidLevel&>(lvl));

    dest.resize(lvl.width*lvl.height);

    dim3 bdim(32,8),
         gdim((lvl.width+bdim.x-1)/bdim.x,
              (lvl.height+bdim.y-1)/bdim.y);

    image_to_internal_vector<T><<<gdim, bdim>>>(&dest, &orig, klvl, mult);
}

template
void image_to_internal_vector(rod::dvector<float> &dest,
                              const rod::dimage<float> &orig,
                              const PyramidLevel &lvl, float mult);

template
void image_to_internal_vector(rod::dvector<float2> &dest,
                              const rod::dimage<float2> &orig,
                              const PyramidLevel &lvl, float2 mult);

