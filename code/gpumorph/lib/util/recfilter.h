#ifndef GPU_RECURSIVE_FILTER_RECFILTER_H
#define GPU_RECURSIVE_FILTER_RECFILTER_H

// Disclaimer: this is optimized for usability, not speed
// Note: everything is row-major (i.e. Vector represents a row in a matrix)

#include <iostream>
#include <cassert>
#include "linalg.h"
#include "dvector.h"
#include "linalg.h"
#include "hostdev.h"

#ifdef  __CUDA_ARCH__
#   ifdef assert
#       undef assert
#   endif
#   define assert (void)
#endif

namespace rod
{

template <class T>
HOSTDEV
T rec_op(const T &x, const T &a)
{
    return x - a;
}

// forward ---------------------------------------------------------

template <class T, int R>
HOSTDEV
T fwd(Vector<T,R> &p, T x, const Vector<T,R+1> &w)
{
    T acc = rec_op(x,p[R-1]*w[1]);
#pragma unroll
    for(int k=R-1; k>=1; --k)
    {
        acc = rec_op(acc,p[R-1-k]*w[k+1]);
        p[R-1-k] = p[R-1-k+1];
    }
    return p[R-1] = acc;
}


template <class T, int N, int R>
void fwd_inplace(const Vector<T,R> &_p, Vector<T,N> &b, const Vector<T,R+1> &w)
{
    Vector<T,R> p = _p;
    for(int j=0; j<b.size(); ++j)
        b[j] = fwd(p, w[0]*b[j], w);
}

template <class T, int M, int N, int R>
void fwdD_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w)
{
    for(int i=0; i<b.rows(); ++i)
        fwd_inplace(p[i], b[i], w);
}

template <class T, int M, int N, int R>
void fwd_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w)
{
    fwdD_inplace(p,b,w);
}

template <class T, int M, int N, int R>
Matrix<T,M,N> fwdD(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, 
              const Vector<T,R+1> &w)
{
    Matrix<T,M,N> fb = b;
    fwdD_inplace(p, fb, w);
    return fb;
}

template <class T, int M, int N, int R>
Matrix<T,M,N> fwd(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, 
              const Vector<T,R+1> &w)
{
    return fwdD(p, b, w);
}

// forward transposed  ----------------------------------------------------

template <class T, int M, int N, int R>
Matrix<T,M,N> fwdT(const Matrix<T,R,N> &pT, const Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w)
{
    return transp(fwd(transp(pT), transp(b), w));
}

template <class T, int M, int N, int R>
Matrix<T,M,N> fwd(const Matrix<T,R,N> &pT, const Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w)
{
    return fwdT(pT, b, w);
}

template <class T, int M, int N, int R>
void fwdT_inplace(const Matrix<T,R,N> &p, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w)
{
    b = fwdT(p, b, w);
}

template <class T, int M, int N, int R>
void fwd_inplace(const Matrix<T,R,N> &p, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w)
{
    fwdT_inplace(p, b, w);
}


// reverse ---------------------------------------------------------------

template <class T, int R>
HOSTDEV
T rev(T x, Vector<T,R> &e, const Vector<T,R+1> &w)
{
    T acc = rec_op(x,e[0]*w[1]);
#pragma unroll
    for(int k=R-1; k>=1; --k)
    {
        acc = rec_op(acc,e[k]*w[k+1]);
        e[k] = e[k-1];
    }
    return e[0] = acc;
}

template <class T, int N, int R>
void rev_inplace(Vector<T,N> &b, const Vector<T,R> &_e, const Vector<T,R+1> &w)
{
    Vector<T,R> e = _e;
    for(int j=b.size()-1; j>=0; --j)
        b[j] = rev(w[0]*b[j], e, w);
}

template <class T, int M, int N, int R>
void revD_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
              const Vector<T,R+1> &w)
{
    for(int i=0; i<b.rows(); ++i)
        rev_inplace(b[i], e[i], w);
}

template <class T, int M, int N, int R>
void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
              const Vector<T,R+1> &w)
{
    revD_inplace(b, e, w);
}

template <class T, int M, int N, int R>
Matrix<T,M,N> revD(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
              const Vector<T,R+1> &w)
{
    Matrix<T,M,N> rb = b;
    revD_inplace(rb, e, w);

    return rb;
}

template <class T, int M, int N, int R>
Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
              const Vector<T,R+1> &w)
{
    return revD(b, e, w);
}

// reverse transposed  ----------------------------------------------------

template <class T, int M, int N, int R>
Matrix<T,M,N> revT(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                  const Vector<T,R+1> &w)
{
    return transp(rev(transp(b), transp(eT), w));
}

template <class T, int M, int N, int R>
Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                  const Vector<T,R+1> &w)
{
    return revT(b,eT,w);
}

template <class T, int M, int N, int R>
void revT_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &p, 
                 const Vector<T,R+1> &w)
{
    b = revT(b, p, w);
}

template <class T, int M, int N, int R>
void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &p, 
                 const Vector<T,R+1> &w)
{
    revT_inplace(b, p, w);
}


// head ---------------------------------------------------------------

template <int R, int M, int N, class T>
Matrix<T,M,R> head(const Matrix<T,M,N> &mat)
{
    assert(mat.cols() >= R);

    Matrix<T,M,R> h;
    for(int j=0; j<R; ++j)
        for(int i=0; i<mat.rows(); ++i)
            h[i][j] = mat[i][j];

    return h;
}

// head transposed  ----------------------------------------------------

template <int R, int M, int N, class T>
Matrix<T,R,N> headT(const Matrix<T,M,N> &mat)
{
    return transp(head<R>(transp(mat)));
}


// tail ---------------------------------------------------------------

template <int R, int M, int N, class T>
Matrix<T,M,R> tail(const Matrix<T,M,N> &mat)
{
    assert(mat.cols() >= R);

    Matrix<T,M,R> t;
    for(int j=0; j<R; ++j)
        for(int i=0; i<mat.rows(); ++i)
            t[i][j] = mat[i][mat.cols()-R+j];

    return t;
}

// tail transposed  ----------------------------------------------------

template <int R, int M, int N, class T>
Matrix<T,R,N> tailT(const Matrix<T,M,N> &mat)
{
    return transp(tail<R>(transp(mat)));
}

enum BorderType
{
    CLAMP_TO_ZERO,
    CLAMP_TO_EDGE,
    REPEAT,
    REFLECT
};

struct recfilter5_plan;

typedef float pixel_type;
template <int R>
recfilter5_plan *recfilter5_create_plan(int width, int height, int rowstride,
                                        const Vector<float, R+1> &w,
                                        BorderType border_type=REFLECT, 
                                        int border=1);

template <int R>
void update_plan(recfilter5_plan *plan, int width, int height, int rowstride,
                 const Vector<float, R+1> &w,
                 BorderType border_type=REFLECT, int border=1);

void free(recfilter5_plan *plan);

void recfilter5(recfilter5_plan *plan, float *d_inout);
void recfilter5(recfilter5_plan *plan, float *d_output, const float *d_input);

}

#endif

