#include <util/dimage.h>
#include <util/image_ops.h>

#define IMAGEMAGICK 1
#define OPENCV 2
#define NEHAB 3

#define METHOD NEHAB

#if METHOD == IMAGEMAGICK

#include <unistd.h> // for unlink
#include <cstdio>
#include <cerrno>
#include <cstring>
#include "imgio.h"

void downsample(rod::dimage<float> &dest, const rod::dimage<float> &orig)
{
    save("orig.png",orig);

    char cmd[128];

    sprintf(cmd,"convert orig.png -resize %dx%d dest.png",
            dest.width(),dest.height());
    if(system(cmd) == -1)
        throw std::runtime_error(strerror(errno));

    load(dest, "dest.png");

    unlink("orig.png");
    unlink("dest.png");
}

#elif METHOD == OPENCV

#include <opencv2/core/wimage.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void downsample(rod::dimage<float> &dest, const rod::dimage<float> &orig)
{
    rod::dimage<unsigned char> imgaux(orig.width(), orig.height());
    convert(&imgaux, &orig, false);

    std::vector<unsigned char> cpu;
    imgaux.copy_to_host(cpu);

    cv::WImageViewC<unsigned char,1> temp(&cpu[0], orig.width(), orig.height());

    cv::Mat out;
    resize((cv::Mat)temp.Ipl(), out, cv::Size(dest.width(),dest.height()),
           0,0,cv::INTER_AREA);

    IplImage temp2 = out;
    cv::WImageView<unsigned char> img = &temp2;

    rod::dimage<uchar1> temp3;
    temp3.copy_from_host(img(0,0), img.Width(),
                        img.Height(), img.WidthStep());
    convert(&dest, &temp3, false);
}

#elif METHOD == NEHAB

#include <resample/scale.h>
#include "imgio.h"

void downsample(rod::dimage<float> &dest, const rod::dimage<float> &orig)
{
    using namespace nehab;

    image::rgba<float> orig_rgba;
    image::load(orig, &orig_rgba);

    kernel::base *pre = new kernel::generalized(
            new kernel::discrete::delta,
            new kernel::discrete::sampled(new kernel::generating::bspline3),
            new kernel::generating::bspline3);
    // no additional discrete processing
    kernel::discrete::base *delta = new kernel::discrete::delta;
    // use mirror extension
    extension::base *ext = new extension::mirror;

    scale(dest.height(), dest.width(), pre, delta, delta, ext, &orig_rgba, dest);

    return;
}


#else
#   error bad downsampling method
#endif
