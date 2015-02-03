#ifndef DLTI_H
#define DLTI_H

#include "image.h"
#include "discrete.h"

namespace dlti {
    // finite impulse-response (i.e., direct convolution)
    int fir(const float *kernel, int W, image::rgba<float> *rgba);
    int fir_rows(const float *kernel, int W, image::rgba<float> *rgba);
    int fir_columns(const float *kernel, int W, image::rgba<float> *rgba);
    int fir(const float *kernel, int W, float *array, int n);
    // inverse of finite impulse-response (i.e., deconvolution)
    int ifir(const float *kernel, int W, image::rgba<float> *rgba);
    int ifir_rows(const float *kernel, int W, image::rgba<float> *rgba);
    int ifir_columns(const float *kernel, int W, image::rgba<float> *rgba);
    int ifir(const float *kernel, int W, float *array, int n);
    // full IIR filtering 
    static inline int filter(const kernel::discrete::base *f, 
        const kernel::discrete::base *i, image::rgba<float> *rgba) {
        return fir(f->v, f->support(), rgba) && 
               ifir(i->v, i->support(), rgba);
    }
    static inline int filter_rows(const kernel::discrete::base *f, 
        const kernel::discrete::base *i, image::rgba<float> *rgba) {
        return fir_rows(f->v, f->support(), rgba) && 
               ifir_rows(i->v, i->support(), rgba);
    }
    static inline int filter_columns(const kernel::discrete::base *f, 
        const kernel::discrete::base *i, image::rgba<float> *rgba) {
        return fir_columns(f->v, f->support(), rgba) && 
               ifir_columns(i->v, i->support(), rgba);
    }
    static inline int filter(const kernel::discrete::base *f, 
        const kernel::discrete::base *i, float *array, int n) {
        return fir(f->v, f->support(), array, n) && 
               ifir(i->v, i->support(), array, n);
    }
}

#endif
