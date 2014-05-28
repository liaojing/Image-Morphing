#ifndef NLFILTER_IMAGE_UTIL_H
#define NLFILTER_IMAGE_UTIL_H

#include <vector>
#include <string>
#include "dimage.h"
#include "util.h"

namespace rod
{

// convert ----------------------------

template <class T, int C, class U, int D>
void convert(dimage_ptr<T,C> out, dimage_ptr<const U,D> in, bool rescale=true);

template <class T, int C, class U, int D>
void convert(dimage_ptr<T,C> out, dimage_ptr<U,D> in, bool rescale=true)
{
    convert(out, dimage_ptr<const U,D>(in), rescale);
}

template <class T, int C>
void convert(dimage_ptr<T,C> out, dimage_ptr<const T,C> in, bool rescale=true)
{
    out = in;
}

// grayscale ----------------------------

template <class T, int C>
void grayscale(dimage_ptr<float> out, dimage_ptr<const T,C> in);

template <class T, int C>
void grayscale(dimage_ptr<float> out, dimage_ptr<T,C> in)
{
    grayscale(out, dimage_ptr<const T,C>(in));
}

// convolve ----------------------------

template <class T, int C, class U, int D, int R>
void convolve(dimage_ptr<T,C> out, dimage_ptr<const U,D> in,
              const array<float,R> &kernel, int scale=1);

template <class T, int C, class U, int D, int R>
void convolve(dimage_ptr<T,C> out, dimage_ptr<U,D> in,
              const array<float,R> &kernel, int scale=1)
{
    convolve(out, dimage_ptr<const U,D>(in), kernel, scale);
}

// lrgb2srgb ----------------------------

template <class T, int C, class U, int D>
void lrgb2srgb(dimage_ptr<T,C> out, dimage_ptr<const U,D> in);

template <class T, int C, class U, int D>
void lrgb2srgb(dimage_ptr<T,C> out, dimage_ptr<U,D> in)
{
    lrgb2srgb(out, dimage_ptr<const U,D>(in));
}

// luminance -----------------------------
//
template <class T, int C>
void luminance(dimage_ptr<float> out, dimage_ptr<const T,C> in);

template <class T, int C>
void luminance(dimage_ptr<float> out, dimage_ptr<T,C> in)
{
    luminance(out, dimage_ptr<const T,C>(in));
}

// gaussian blur ------------------------------------------

struct gaussian_blur_plan;

gaussian_blur_plan *gaussian_blur_create_plan(int width, int height,
                                              int rowstride, float sigma);
void free(gaussian_blur_plan *);

void update_plan(gaussian_blur_plan *plan, int width, int height, 
                 int rowstride, float sigma);

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out, 
                   dimage_ptr<const float,C> in);

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out, 
                   dimage_ptr<float,C> in)
{
    gaussian_blur(plan, out, dimage_ptr<const float,C>(in));
}

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out)
{
    gaussian_blur(plan, out, dimage_ptr<const float,C>(out));
}

// up/downsample ----------------------------------------------

enum Interpolation
{
    INTERP_BOX,
    INTERP_LINEAR,
    INTERP_BSPLINE3,
    INTERP_MITCHELL_NETRAVALI
};

template <class T, int C>
void upsample(dimage_ptr<T,C> out, dimage_ptr<const T,C> in, 
              Interpolation interp = INTERP_LINEAR);
template <class T, int C>
void upsample(dimage_ptr<T,C> out, dimage_ptr<T,C> in, 
              Interpolation interp = INTERP_LINEAR)
{
    upsample(out, dimage_ptr<const T,C>(in), interp);
}

template <class T, int C>
void downsample(dimage_ptr<T,C> out, dimage_ptr<const T,C> in, 
              Interpolation interp = INTERP_MITCHELL_NETRAVALI);
template <class T, int C>
void downsample(dimage_ptr<T,C> out, dimage_ptr<T,C> in, 
              Interpolation interp = INTERP_MITCHELL_NETRAVALI)
{
    downsample(out, dimage_ptr<const T,C>(in), interp);
}

template <class T, int C>
void downsample(dimage_ptr<T,C> out, const cudaArray *in, size_t w, size_t h, 
              Interpolation interp = INTERP_MITCHELL_NETRAVALI);


} // namespace rod

#endif
