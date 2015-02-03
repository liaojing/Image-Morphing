#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "extension.h"
#include "discrete.h"
#include "error.h"
#include "dlti.h"

static const float EPS = 1.e-6f;

// simple banded matrix class
class matrix {
public:
    matrix(void) {
        data = NULL; 
        M = N = 0;
    }
    ~matrix() {
        free(data);
    }
    int resize(int m, int n) {
        float *tmp = (float *) realloc(data, sizeof(float)*m*n);
        if (!tmp) {
            warnf(("out of memory"));
            return 0;
        } else {
            data = tmp;
            N = n;
            M = m;
            return 1;
        }
    }
    // index banded representation as if it was flat
    const float &operator()(int i, int j) const {
        return data[(i-j+M/2)*N+j];
    }
    float &operator()(int i, int j) {
        return data[(i-j+M/2)*N+j];
    }
    int size(void) const { return N; }
    int bandwidth(void) const { return M; }
    void dump(void) const {
        const matrix &A = (*this);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (abs(j-i) <= M/2)
                    fprintf(stderr, "%9.6g ", A(i,j));
                else
                    fprintf(stderr, "%9.6g ", 0.f);
            }
            fprintf(stderr, "\n");
        }
    }
    void clear(float f = 0.f) {
        for (int i = 0; i < M*N; i++) {
            data[i] = f;
        }
    }
private:
    matrix(const matrix& rhs);
    matrix& operator=(const matrix& rhs);
    float *data;
    int N, M;
};

// performs in-place LU decomposition of a banded matrix
static int factor(matrix *pA) {
    matrix &A = (*pA);
    if (A.bandwidth() % 2 == 0) {
        warnf(("matrix bandwidth is even"));
        return 0;
    }
    int r = A.bandwidth()/2;
    // for each pivot
    for (int p = 0; p < A.size(); p++) {
        // factorization failed? 
        if (fabs(A(p,p)) < EPS) {
            warnf(("matrix singular or pivotting needed"));
            return 0;
        }
        float inv_p = (A(p,p) = 1.f/A(p,p));
        // for each row below the pivot
        for (int i = p+1; i <= p+r && i < A.size(); i++) {
            // perform row operation that kills column below pivot
            float m = (A(i,p) *= inv_p);
            for (int j = p+1; j <= p+r && j < A.size(); j++) {
                A(i,j) -= m*A(p,j);
            }
        }
    }
    return 1;
}

// forward- and back-substitute on all rows in an image in-place
// receives a factored matrix!
static int solve_rows(const matrix &A, float *channel, int h, int w) {
    if (A.size() != w) {
        warnf(("matrix size %d incompatible with %d columns in rhs", 
            A.size(), w));
        return 0;
    }
    if (A.bandwidth() % 2 == 0) {
        warnf(("matrix bandwidth is even"));
        return 0;
    }
    int r = A.bandwidth()/2;
#pragma omp parallel for
    for (int i = 0; i < h; i++) {
        // forward-substitute
        for (int j = 0; j < w; j++) {
            for (int k = r; k > 0; k--) {
                if ((j-k) >= 0)
                    channel[i*w+j] -= A(j,j-k)*channel[i*w+j-k];
            }
        }
        // back-substitute
        for (int j = w-1; j >= 0; j--) {
            for (int k = r; k > 0; k--) {
                if ((j+k) < w) 
                    channel[i*w+j] -= A(j,j+k)*channel[i*w+j+k];
            }
            channel[i*w+j] *= A(j,j);
        }
    }
    return 1;
}

// forward- and back-substitute on all columns in an image
// inverted loop is faster because of caching
// receives a factored matrix!
static int solve_columns(const matrix &A, float *channel, int h, int w) {
    if (A.size() != h) {
        warnf(("matrix size %d incompatible with %d rows in rhs", 
            A.size(), h));
        return 0;
    }
    if (A.bandwidth() % 2 == 0) {
        warnf(("matrix bandwidth is even"));
        return 0;
    }
    int r = A.bandwidth()/2;
    // forward-substitute 
    for (int i = 0; i < h; i++) {
        for (int k = r; k > 0; k--) {
            if ((i-k) >= 0) {
// I don't know why this causes an abort trap every now and then. :(
// #pragma omp parallel for
                for (int j = 0; j < w; j++) {
                    channel[i*w+j] -= A(i,i-k)*channel[(i-k)*w+j];
                }
            }
        }
    }
    // back-substitute
    for (int i = h-1; i >= 0; i--) {
        for (int k = r; k > 0; k--) {
            if ((i+k) < h) {
// I don't know why this causes an abort trap every now and then. :(
// #pragma omp parallel for
                for (int j = 0; j < w; j++) {
                    channel[i*w+j] -= A(i,i+k)*channel[(i+k)*w+j];
                }
            }
        }
        for (int j = 0; j < w; j++)
            channel[i*w+j] *= A(i,i);
    }
    return 1;
}

static float *convolve_cache(int W, int w) {
    // allocate cache so we can convolve in-place
    // otherwise we would overwrite stuff we still need!
    return static_cast<float *>(malloc(sizeof(float)*W*w));
}

// convolve all image rows with kernel 
static int convolve_rows(const float *kernel, int W, 
    float *cache, float *channel, int h, int w) {
    extension::mirror mirror;
    if (W % 2 == 0) {
        warnf(("kernel support is even"));
        return 0;
    }
    int r = W/2;
    // horizontal separable pass
    for (int i = 0; i < h; i++) {
        // copy row to cache
        memcpy(cache, channel+i*w, sizeof(float)*w);
        // read from cache, write to channel 
        for (int j = 0; j < w; j++) {
            float acc = 0.f;
            for (int k = 0; k < W; k++) {
                acc += kernel[k]*cache[mirror(j+k-r, w)];
            }
            channel[i*w+j] = acc;
        }
    }
    return 1;
}

// convolve all image columns with kernel 
static int convolve_columns(const float *kernel, int W, 
    float *cache, float *channel, int h, int w) {
    extension::mirror mirror;
    if (W % 2 == 0) {
        warnf(("kernel support is even"));
        return 0;
    }
    int r = W/2;
    // vertical separable pass
    // prefill cache with W channel rows
    extension::repeat repeat;
    for (int k = 0; k < W; k++)
        memcpy(cache+k*w, channel+mirror(k-r,h)*w, sizeof(float)*w);
    // output each output row 
    for (int i = 0; i < h; i++) {
        // convolve reading from cache and writing to channel row
        for (int j = 0; j < w; j++) {
            float acc = 0.f;
            for (int k = 0; k < W; k++) {
                acc += kernel[repeat(k-i,W)]*cache[k*w+j];
            } 
            channel[i*w+j] = acc;
        }
        // copy next channel row to cache
        memcpy(cache+repeat(i,W)*w, 
            channel+mirror(i+W-r,h)*w, sizeof(float)*w);
    }
    return 1;
}

namespace dlti {

    int ifir_rows(const float *kernel, int W, image::rgba<float> *rgba) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        const int h = rgba->height();
        const int w = rgba->width();
        if (W % 2 == 0) {
            warnf(("kernel support is even"));
            return 0;
        }
        // allocate and zero-out banded matrices
        matrix R;
        if (!R.resize(W, w)) {
            warnf(("out of memory"));
            return 0;
        }
        R.clear(); 
        // fill matrices with convolution operator
        extension::mirror mirror;
        int r = W/2;
        for (int i = 0; i < w; i++) {
            for (int k = 0; k < W; k++) {
                R(i, mirror(i+k-r, w)) += kernel[k];
            }
        }
        // factor matrices into LU
        if (!factor(&R)) {
            warnf(("factoring failed"));
            return 0;
        }
        // solve rows then columns for all channels
        if (!solve_rows(R, rgba->r, h, w) ||
            !solve_rows(R, rgba->g, h, w) ||
            !solve_rows(R, rgba->b, h, w) ||
            !solve_rows(R, rgba->a, h, w)) { 
            warnf(("filtering failed"));
            return 0;
        }
        return 1;
    }

    int ifir_columns(const float *kernel, int W, image::rgba<float> *rgba) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        const int h = rgba->height();
        const int w = rgba->width();
        if (W % 2 == 0) {
            warnf(("kernel support is even"));
            return 0;
        }
        // allocate and zero-out banded matrices
        matrix C;
        if (!C.resize(W, h)) {
            warnf(("out of memory"));
            return 0;
        }
        C.clear();
        // fill matrices with convolution operator
        extension::mirror mirror;
        int r = W/2;
        for (int i = 0; i < h; i++) {
            for (int k = 0; k < W; k++) {
                C(i, mirror(i+k-r, h)) += kernel[k];
            }
        }
        // factor matrices into LU
        if (!factor(&C)) {
            warnf(("factoring failed"));
            return 0;
        }
        // solve rows then columns for all channels
        if (!solve_columns(C, rgba->r, h, w) ||
            !solve_columns(C, rgba->g, h, w) ||
            !solve_columns(C, rgba->b, h, w) ||
            !solve_columns(C, rgba->a, h, w)) {
            warnf(("filtering failed"));
            return 0;
        }
        return 1;
    }

    // inverse convolution
    int ifir(const float *kernel, int W, image::rgba<float> *rgba) {
        return ifir_rows(kernel, W, rgba) && ifir_columns(kernel, W, rgba);
    }

    int ifir(const float *kernel, int W, float *array, int n) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        if (W % 2 == 0) {
            warnf(("kernel support is even"));
            return 0;
        }
        // allocate and zero-out banded matrices
        matrix R;
        if (!R.resize(W, n)) {
            warnf(("out of memory"));
            return 0;
        }
        R.clear();
        // fill matrices with convolution operator
        extension::mirror mirror;
        int r = W/2;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < W; k++) {
                R(i, mirror(i+k-r, n)) += kernel[k];
            }
        }
        // factor matrices into LU
        if (!factor(&R)) {
            warnf(("factoring failed"));
            return 0;
        }
        // solve rows then columns for all channels
        if (!solve_rows(R, array, 1, n)) {
            warnf(("filtering failed"));
            return 0;
        }
        return 1;
    }

    int fir_columns(const float *kernel, int W, image::rgba<float> *rgba) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        const int h = rgba->height();
        const int w = rgba->width();
        float *cache = convolve_cache(W, w);
        if (!cache) {
            warnf(("out of memory"));
            return 0;
        }
        if (!convolve_columns(kernel, W, cache, rgba->r, h, w) ||
            !convolve_columns(kernel, W, cache, rgba->g, h, w) ||
            !convolve_columns(kernel, W, cache, rgba->b, h, w) ||
            !convolve_columns(kernel, W, cache, rgba->a, h, w)) {
            warnf(("filtering failed"));
            return 0;
        }
        free(cache);
        return 1;
    }

    int fir_rows(const float *kernel, int W, image::rgba<float> *rgba) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        const int h = rgba->height();
        const int w = rgba->width();
        float *cache = convolve_cache(W, w);
        if (!cache) {
            warnf(("out of memory"));
            return 0;
        }
        if (!convolve_rows(kernel, W, cache, rgba->r, h, w) ||
            !convolve_rows(kernel, W, cache, rgba->g, h, w) ||
            !convolve_rows(kernel, W, cache, rgba->b, h, w) ||
            !convolve_rows(kernel, W, cache, rgba->a, h, w)) {
            warnf(("filtering failed"));
            return 0;
        }
        free(cache);
        return 1;
    }

    int fir(const float *kernel, int W, image::rgba<float> *rgba) {
        return fir_rows(kernel, W, rgba) && fir_columns(kernel, W, rgba); 
    }

    int fir(const float *kernel, int W, float *array, int n) {
        // optimize for identity case
        if (kernel::discrete::isdelta(kernel, W)) return 1;
        float *cache = convolve_cache(1, n);
        if (!cache) {
            warnf(("out of memory"));
            return 0;
        }
        if (!convolve_rows(kernel, W, cache, array, 1, n)) {
            warnf(("filtering failed"));
            return 0;
        }
        if (!convolve_columns(kernel, W, cache, array, n, 1)) {
            warnf(("filtering failed"));
            return 0;
        }
        free(cache);
        return 1;
    }
}
