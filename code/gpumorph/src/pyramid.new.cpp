#include <cmath>
#include <util/timer.h>
#include <util/dimage.h>
#include <util/image_ops.h>
#include <iostream>
#include <sstream>
#include "parameters.h"
#include "pyramid.h"

Pyramid::~Pyramid()
{
    for(int i=0; i<m_data.size(); ++i)
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
    blockdim.x = (w+4)/5;
    blockdim.y = (h+4)/5;
    // align on 128 byte boundary
    pixstride = (blockdim.x*blockdim.y + 31)/32 * 32;
    inv_wh = 1.0f/(w*h);

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&img0, &ccd, w, h);
    cudaMallocArray(&img1, &ccd, w, h);

    size_t size = pixstride*25;

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

    improving_mask.resize(size);
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

    blockdim = lvl.blockdim;
    pixstride = lvl.pixstride;

    pixdim = make_int2(lvl.width, lvl.height);
    inv_wh = lvl.inv_wh;
}


template <class T> 
T log2(const T &v)/*{{{*/
{
    using std::log;
    return log(v)/log(2.0);
}/*}}}*/

void create_pyramid(const Parameters &params, Pyramid &pyr,
                    const rod::dimage<float3> &img0,
                    const rod::dimage<float3> &img1)
{
    rod::base_timer &timer_total = rod::timers.gpu_add("Pyramid creation");

    size_t nlevels 
        = log2(std::min(img0.width(),img0.height())) - log2(params.start_res)+1;

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

    int base_level = 0;

    for(int l=1; l<nlevels; ++l)
    {
        int w = round(img0.width()/((float)(1<<l))),
            h = round(img0.height()/((float)(1<<l)));

        std::clog << "Level " << l << ": " << w << "x" << h << std::endl;

        std::ostringstream ss;
        ss << "level " << l;

        rod::scoped_timer_stop sts(rod::timers.gpu_add(ss.str(), w*h,"P"));

        PyramidLevel &lvl = pyr.append_new(w,h);

        while((float)pyr[base_level].width/w * pyr[base_level].height/h>=1024)
            ++base_level;

        luma0.resize(w,h);
        luma1.resize(w,h);

        downsample(&luma0, pyr[base_level].img0, 
                   pyr[base_level].width, pyr[base_level].height);
        downsample(&luma1, pyr[base_level].img1,
                   pyr[base_level].width, pyr[base_level].height);

        copy_to_array(lvl.img0, &luma0);
        copy_to_array(lvl.img1, &luma1);
    }

    timer_total.stop();
}
