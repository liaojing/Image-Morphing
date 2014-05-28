#include <omp.h>

#include "kernel.h"
#include "extension.h"
#include "dlti.h"

#define VERBOSE 0

namespace nehab
{

static int upsample_rows(image::rgba<float> *input, int wout,
    const kernel::base *pre, const extension::base *ext,
    image::rgba<float> *output) {

static extension::clamp clamp;

    int hin, win; // input dimensions
    input->size(&hin, &win);
    if (!output->resize(hin, wout)) {
        warnf(("unable to allocate output image"));
        return 0;
    }
    // move image back to gamma space
    input->lrgb2srgb();
    // apply digital prefilter
    if (!dlti::filter_rows(pre->fir(), pre->ifir(), input)) {
        warnf(("failed to apply digital prefilter"));
        return 0;
    }
#if VERBOSE
    fprintf(stderr, "  upsampling rows from %d to %d\n", win, wout);
    fprintf(stderr, "    with delta as prefilter and %s reconstruction\n",
        pre->name());
#endif
    float inv_wout = 1.f/static_cast<float>(wout);
    float inv_sw = static_cast<float>(win)*inv_wout;
    float r = (pre->support()+1)/2;
#pragma omp parallel for
    for (int iout = 0; iout < hin; iout++) {
#if VERBOSE
        if (iout % 13 == 0) {
            fprintf(stderr, "%d ", iout);
            fflush(stderr);
        }
#endif
        for (int jout = 0; jout < wout; jout++) {
            float fjin = (static_cast<float>(jout)+.5f)*inv_sw-.5f;
            int cjin = static_cast<int>(floorf(fjin));
            float djin = fjin-cjin;
            float sum_r = 0.f, sum_g = 0.f,
                  sum_b = 0.f, sum_a = 0.f,
                  sum_w = 0.f;
            for (int j = -r+1; j <= r; j++) {
                float w = pre->value(djin-j);
                int q = iout*win+clamp.wrap(ext->wrap(cjin+j, win), win);
                sum_r += input->r[q]*w; sum_g += input->g[q]*w;
                sum_b += input->b[q]*w; sum_a += input->a[q]*w;
                sum_w += w;
            }
            int p = iout*wout+jout;
            output->r[p] = sum_r; output->g[p] = sum_g;
            output->b[p] = sum_b; output->a[p] = sum_a;
            if (pre->normalize()) {
                output->r[p] /= sum_w; output->g[p] /= sum_w;
                output->b[p] /= sum_w; output->a[p] /= sum_w;
            }
        }
    }
    // move output back to linear
    output->srgb2lrgb();
#if VERBOSE
    fprintf(stderr, "\n");
#endif
    return 1;
}

// we should invert this loop for efficiency
static int upsample_columns(image::rgba<float> *input, int hout,
    const kernel::base *pre, const extension::base *ext,
    image::rgba<float> *output) {

static extension::clamp clamp;

    int hin, win; // input dimensions
    input->size(&hin, &win);
    if (!output->resize(hout, win)) {
        warnf(("unable to allocate output image"));
        return 0;
    }
    // move image back to gamma space
    input->lrgb2srgb();
    // apply digital prefilter
    if (!dlti::filter_columns(pre->fir(), pre->ifir(), input)) {
        warnf(("failed to apply digital prefilter"));
        return 0;
    }
#if VERBOSE
    fprintf(stderr, "  upsampling columns from %d to %d\n", hin, hout);
    fprintf(stderr, "    with delta as prefilter and %s reconstruction\n",
        pre->name());
#endif
    float inv_hout = 1.f/static_cast<float>(hout);
    float inv_sw = static_cast<float>(hin)*inv_hout;
    float r = (pre->support()+1)/2;
#pragma omp parallel for
    for (int jout = 0; jout < win; jout++) {
#if VERBOSE
        if (jout % 13 == 0) {
            fprintf(stderr, "%d ", jout);
            fflush(stderr);
        }
#endif
        for (int iout = 0; iout < hout; iout++) {
            float fiin = (static_cast<float>(iout)+.5f)*inv_sw-.5f;
            int ciin = static_cast<int>(floorf(fiin));
            float diin = fiin-ciin;
            float sum_r = 0.f, sum_g = 0.f,
                  sum_b = 0.f, sum_a = 0.f,
                  sum_w = 0.f;
            for (int i = -r+1; i <= r; i++) {
                float w = pre->value(diin-i);
                int q = clamp.wrap(ext->wrap(ciin+i, hin), hin)*win + jout;
                sum_r += input->r[q]*w; sum_g += input->g[q]*w;
                sum_b += input->b[q]*w; sum_a += input->a[q]*w;
                sum_w += w;
            }
            int p = iout*win+jout;
            output->r[p] = sum_r; output->g[p] = sum_g;
            output->b[p] = sum_b; output->a[p] = sum_a;
            if (pre->normalize()) {
                output->r[p] /= sum_w; output->g[p] /= sum_w;
                output->b[p] /= sum_w; output->a[p] /= sum_w;
            }
        }
    }
    // move output back to linear
    output->srgb2lrgb();
#if VERBOSE
    fprintf(stderr, "\n");
#endif
    return 1;
}

// we should invert this loop for efficiency
static int downsample_columns(const image::rgba<float> &input, int hout,
    const kernel::base *pre, const extension::base *ext,
    image::rgba<float> *output) {

static extension::clamp clamp;

    int hin, win; // input dimensions
    input.size(&hin, &win);
    if (!output->resize(hout, win)) {
        warnf(("unable to allocate output image"));
        return 0;
    }
#if VERBOSE
    fprintf(stderr, "  downsampling columns from %d to %d\n", hin, hout);
    fprintf(stderr, "    with %s prefilter and delta reconstruction\n",
        pre->name());
#endif
    float inv_hin = 1.f/static_cast<float>(hin);
    float inv_sw = static_cast<float>(hout)*inv_hin;
    float sw = 1.f/inv_sw;
    float s = pre->support();
#pragma omp parallel for
    for (int jout = 0; jout < win; jout++) {
#if VERBOSE
        if (jout % 13 == 0) {
            fprintf(stderr, "%d ", jout);
            fflush(stderr);
        }
#endif
        for (int iout = 0; iout < hout; iout++) {
            int min_iin = (int) ceilf(.5f*sw*(2.f*iout+1.f-s)-.5f);
            int max_iin = (int) floorf(.5f*sw*(2.f*iout+1.f+s)-.5f);
            if (min_iin > max_iin) // just to work with delta
                min_iin = max_iin = (int)(.5f*sw*(2.f*iout+1.f));
            float sum_r = 0.f, sum_g = 0.f,
                  sum_b = 0.f, sum_a = 0.f,
                  sum_w = 0.f;
            for (int iin = min_iin; iin <= max_iin; iin++) {
                float kj = 0.5+iout-(iin+0.5f)*inv_sw;
                float w = pre->value(kj);
                int q = clamp.wrap(ext->wrap(iin, hin), hin)*win+jout;
                float r = input.r[q], g = input.g[q],
                      b = input.b[q], a = input.a[q];
                sum_r += r*w; sum_g += g*w;
                sum_b += b*w; sum_a += a*w;
                sum_w += w;
            }
            // always normalize before saving
            int p = iout*win+jout;
            output->r[p] = sum_r/sum_w; output->g[p] = sum_g/sum_w;
            output->b[p] = sum_b/sum_w; output->a[p] = sum_a/sum_w;
        }
    }
#if VERBOSE
    fprintf(stderr, "\n");
#endif
    return dlti::filter_columns(pre->fir(), pre->ifir(), output);
}

static int downsample_rows(const image::rgba<float> &input, int wout,
    const kernel::base *pre, const extension::base *ext,
    image::rgba<float> *output) {

    static extension::clamp clamp;

    int hin, win; // input dimensions
    input.size(&hin, &win);
    if (!output->resize(hin, wout)) {
        warnf(("unable to allocate output image"));
        return 0;
    }
#if VERBOSE
    fprintf(stderr, "  downsampling rows from %d to %d\n", win, wout);
    fprintf(stderr, "    with %s prefilter and delta reconstruction\n",
        pre->name());
#endif
    float inv_win = 1.f/static_cast<float>(win);
    float inv_sw = static_cast<float>(wout)*inv_win;
    float sw = 1.f/inv_sw;
    float s = pre->support();
#pragma omp parallel for
    for (int iout = 0; iout < hin; iout++) {
#if VERBOSE
        if (iout % 13 == 0) {
            fprintf(stderr, "%d ", iout);
            fflush(stderr);
        }
#endif
        for (int jout = 0; jout < wout; jout++) {
            int min_jin = (int) ceilf(.5f*sw*(2.f*jout+1.f-s)-.5f);
            int max_jin = (int) floorf(.5f*sw*(2.f*jout+1.f+s)-.5f);
            if (min_jin > max_jin) // just to work with delta
                min_jin = max_jin = (int)(.5f*sw*(2.f*jout+1.f));
            float sum_r = 0.f, sum_g = 0.f,
                  sum_b = 0.f, sum_a = 0.f,
                  sum_w = 0.f;
            for (int jin = min_jin; jin <= max_jin; jin++) {
                float kj = 0.5+jout-(jin+0.5f)*inv_sw;
                float w = pre->value(kj);
                int q = iout*win+clamp.wrap(ext->wrap(jin, win), win);
                float r = input.r[q], g = input.g[q],
                      b = input.b[q], a = input.a[q];
                sum_r += r*w; sum_g += g*w;
                sum_b += b*w; sum_a += a*w;
                sum_w += w;
            }
            // always normalize before saving
            int p = iout*wout+jout;
            output->r[p] = sum_r/sum_w; output->g[p] = sum_g/sum_w;
            output->b[p] = sum_b/sum_w; output->a[p] = sum_a/sum_w;
        }
    }
#if VERBOSE
    fprintf(stderr, "\n");
#endif
    return dlti::filter_rows(pre->fir(), pre->ifir(), output);
}

int scale(int hout, int wout, const kernel::base *pre,
    const kernel::discrete::base *fir, const kernel::discrete::base *ifir,
    const extension::base *ext, image::rgba<float> *rgba, rod::dimage<float> &output)
{
    int hin, win;
    rgba->size(&hin, &win);
    image::rgba<float> temp;
    if (hout*win < wout*hin) {
        if (hout < hin) {
            if (!downsample_columns(*rgba, hout, pre, ext, &temp))
                errorf(("error downsampling columns"));
        } else {
            if (!upsample_columns(rgba, hout, pre, ext, &temp))
                errorf(("error upsampling columns"));
        }
        if (wout < win) {
            if (!downsample_rows(temp, wout, pre, ext, rgba))
                errorf(("error downsampling rows"));
        } else {
            if (!upsample_rows(&temp, wout, pre, ext, rgba))
                errorf(("error upsampling rows"));
        }
    } else {
        if (wout < win) {
            if (!downsample_rows(*rgba, wout, pre, ext, &temp))
                errorf(("error downsampling rows"));
        } else {
            if (!upsample_rows(rgba, wout, pre, ext, &temp))
                errorf(("error upsampling rows"));
        }
        if (hout < hin) {
            if (!downsample_columns(temp, hout, pre, ext, rgba))
                errorf(("error downsampling columns"));
        } else {
            if (!upsample_columns(&temp, hout, pre, ext, rgba))
                errorf(("error upsampling columns"));
        }
    }

    if (!dlti::filter(fir, ifir, rgba)) {
        errorf(("discrete filtering failed"));
    }


     if (!image::store(output, (*rgba)))
            errorf(("error saving image"));

    return 1;
}

}
