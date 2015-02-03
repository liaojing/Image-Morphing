#include <iostream>
#include <cassert>
#include <stdexcept>
#include <math.h>
#include <algorithm>
#if HAS_GSL
#   include <complex>
#   include <gsl/gsl_poly.h>
#endif

#ifdef  __CUDA_ARCH__
#   ifdef assert
#       undef assert
#   endif
#   define assert (void)
#elif defined _MSC_VER
#	pragma warning(push)
#	pragma warning(disable:4068) // unknown pragma
#endif

namespace rod
{

// Disclaimer: this is optimized for usability, not speed
// Note: everything is row-major (i.e. Vector represents a row in a matrix)

namespace detail
{
#ifdef __CUDA_ARCH__
    template<class T>
    __device__ void swap(T &a, T &b)
    {
        T c = a;
        a = b;
        b = c;
    }
#else
    using std::swap;
#endif
}

// Vector<T,N> -------------------------------------------------------

template <class T, int N>
std::vector<T> Vector<T,N>::to_vector() const
{
    return std::vector<T>(&m_data[0], &m_data[0]+size());
}

template <class T, int N>
HOSTDEV 
const T &Vector<T,N>::operator[](int i) const
{
    assert(i >= 0 && i < size());
    return m_data[i];
}

template <class T, int N>
HOSTDEV 
T &Vector<T,N>::operator[](int i)
{
    assert(i >= 0 && i < size());
    return m_data[i];
}

template <class T, int N>
HOSTDEV 
Vector<T,N> &operator+=(Vector<T,N> &lhs, const Vector<T,N> &rhs)
{
#pragma unroll
    for(int j=0; j<N; ++j)
        lhs[j] += rhs[j];

    return lhs;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> operator+(const Vector<T,N> &a, const Vector<T,N> &b)
{
    Vector<T,N> r(a);
    return r += b;
}


template <class T, int N>
HOSTDEV 
Vector<T,N> &operator*=(Vector<T,N> &lhs, const T &rhs)
{
#pragma unroll
    for(int j=0; j<lhs.size(); ++j)
        lhs[j] *= rhs;

    return lhs;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> operator*(const Vector<T,N> &a, const T &b)
{
    Vector<T,N> r(a);
    return r *= b;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> operator*(const T &a, const Vector<T,N> &b)
{
    return b*a;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> &operator/=(Vector<T,N> &lhs, const T &rhs)
{
#pragma unroll
    for(int j=0; j<N; ++j)
        lhs[j] /= rhs;

    return lhs;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> operator/(const Vector<T,N> &a, const T &b)
{
    Vector<T,N> r(a);
    return r /= b;
}

template <class T, int N>
std::ostream &operator<<(std::ostream &out, const Vector<T,N> &v)
{
    out << '[';
    for(int i=0; i<v.size(); ++i)
    {
        out << v[i];
        if(i < v.size()-1)
            out << ',';
    }
    return out << ']';
}

template <class T, int N>
Vector<T,N> operator-(const Vector<T,N> &v)
{
    Vector<T,N> r;
    for(int i=0; i<N; ++i)
        r[i] = -v[i];
    return r;
}


// Matrix<T,M,N> -------------------------------------------------------

template <class T, int M, int N>
HOSTDEV const Vector<T,N> &Matrix<T,M,N>::operator[](int i) const
{
    assert(i >= 0 && i < rows());
    return m_data[i];
}

template <class T, int M, int N>
HOSTDEV Vector<T,N> &Matrix<T,M,N>::operator[](int i)
{
    assert(i >= 0 && i < rows());
    return m_data[i];
}

template <class T, int M, int N>
HOSTDEV 
Vector<T,M> Matrix<T,M,N>::col(int j) const
{
    Vector<T,M> c;
#pragma unroll
    for(int i=0; i<rows(); ++i)
        c[i] = m_data[i][j];
    return c;
}

template <class T, int M, int N>
HOSTDEV 
void Matrix<T,M,N>::set_col(int j, const Vector<T,M> &c)
{
#pragma unroll
    for(int i=0; i<rows(); ++i)
        m_data[i][j] = c[i];
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> operator*(const Matrix<T,M,N> &m, T val)
{
    Matrix<T,M,N> r(m);
    return r *= val;
}


template <class T, int M, int N>
std::ostream &operator<<(std::ostream &out, const Matrix<T,M,N> &m)
{
    out << '[';
    for(int i=0; i<m.rows(); ++i)
    {
        for(int j=0; j<m.cols(); ++j)
        {
            out << m[i][j];
            if(j < m.cols()-1)
                out << ',';
        }
        if(i < m.rows()-1)
            out << ";\n";
    }
    return out << ']';
}

template <class T, int M, int N, int P>
HOSTDEV 
Matrix<T,M,P> operator*(const Matrix<T,M,N> &a, const Matrix<T,N,P> &b)
{
    Matrix<T,M,P> r;
#pragma unroll
    for(int i=0; i<M; ++i)
    {
#pragma unroll
        for(int j=0; j<P; ++j)
        {
            r[i][j] = a[i][0]*b[0][j];
#pragma unroll
            for(int k=1; k<N; ++k)
                r[i][j] += a[i][k]*b[k][j];
        }
    }
    return r;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> &operator*=(Matrix<T,M,N> &lhs, T rhs)
{
#pragma unroll
    for(int i=0; i<lhs.rows(); ++i)
#pragma unroll
        for(int j=0; j<lhs.cols(); ++j)
            lhs[i][j] *= rhs;
    return lhs;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> &operator/=(Matrix<T,M,N> &lhs, T rhs)
{
#pragma unroll
    for(int i=0; i<lhs.rows(); ++i)
#pragma unroll
        for(int j=0; j<lhs.cols(); ++j)
            lhs[i][j] /= rhs;
    return lhs;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> operator*(T val, const Matrix<T,M,N> &m)
{
    return m * val;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> &operator+=(Matrix<T,M,N> &lhs, const Matrix<T,M,N> &rhs)
{
#pragma unroll
    for(int i=0; i<M; ++i)
#pragma unroll
        for(int j=0; j<N; ++j)
            lhs[i][j] += rhs[i][j];
    return lhs;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> operator+(const Matrix<T,M,N> &lhs, const Matrix<T,M,N> &rhs)
{
    Matrix<T,M,N> r(lhs);
    return r += rhs;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> &operator-=(Matrix<T,M,N> &lhs, const Matrix<T,M,N> &rhs)
{
#pragma unroll
    for(int i=0; i<M; ++i)
#pragma unroll
        for(int j=0; j<N; ++j)
            lhs[i][j] -= rhs[i][j];
    return lhs;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> operator-(const Matrix<T,M,N> &a, const Matrix<T,M,N> &b)
{
    Matrix<T,M,N> r(a);
    return r -= b;
}


template <class T, int M, int N>
HOSTDEV 
Vector<T,N> operator*(const Vector<T,M> &v, const Matrix<T,M,N> &m)
{
    Vector<T,N> r;

#pragma unroll
    for(int j=0; j<m.cols(); ++j)
    {
        r[j] = v[0]*m[0][j];
#pragma unroll
        for(int i=1; i<m.rows(); ++i)
            r[j] += v[i]*m[i][j];
    }

    return r;
}

template <class T, int M, int N>
HOSTDEV 
Vector<T,N> &operator*=(Vector<T,M> &v, const Matrix<T,M,N> &m)
{
    return v = v*m;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> &operator*=(Matrix<T,M,N> &lhs, const Matrix<T,N,N> &rhs)
{
    return lhs = lhs*rhs;
}

template <class T, int M, int N>
HOSTDEV
Matrix<T,M,N> operator-(const Matrix<T,M,N> &m)
{
    Matrix<T,M,N> r;
    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            r[i][j] = -m[i][j];
    return r;
}


// Special matrices -------------------------------------------------------

template <class T, int N>
HOSTDEV 
Vector<T,N> zeros()
{
    Vector<T,N> v;
    if(N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for(int j=0; j<v.size(); ++j)
            v[j] = T();
#else
	// T() is failing to return a zero-initialized value on
	// VC2010, better use memset for the time being.
	memset(&mat, 0, sizeof(mat));
        //std::fill(&v[0], &v[N-1]+1, T());
#endif
    }
    return v; // I'm hoping that RVO will kick in
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> zeros()
{
    Matrix<T,M,N> mat;
    if(M>0 && N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for(int i=0; i<mat.rows(); ++i)
#pragma unroll
            for(int j=0; j<mat.cols(); ++j)
                mat[i][j] = T();
#else
	// T() is failing to return a zero-initialized value on
	// VC2010, better use memset for the time being.
	memset(&mat, 0, sizeof(mat));
        //std::fill(&mat[0][0], &mat[M-1][N-1]+1, T());
#endif
    }
    return mat; // I'm hoping that RVO will kick in
}


template <class T, int N>
HOSTDEV 
Vector<T,N> ones()
{
    Vector<T,N> v;
    if(N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for(int j=0; j<v.size(); ++j)
            v[j] = T(1);
#else
        std::fill(&v[0], &v[N-1]+1, T(1));
#endif
    }
    return v; // I'm hoping that RVO will kick in
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> ones()
{
    Matrix<T,M,N> mat;
    if(M>0 && N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for(int i=0; i<mat.rows(); ++i)
#pragma unroll
            for(int j=0; j<mat.cols(); ++j)
                mat[i][j] = T(1);
#else
        std::fill(&mat[0][0], &mat[M-1][N-1]+1, T(1));
#endif
    }
    return mat; // I'm hoping that RVO will kick in
}

template <class T, int M, int N>
Matrix<T,M,N> identity()
{
    Matrix<T,M,N> mat;

    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            mat[i][j] = i==j ? T(1) : T(0);

    return mat;
}

template <class T, int M>
Matrix<T,M,M> vander(const Vector<T,M> &v)
{
    using std::pow;

    Matrix<T,M,M> m;
    for(int i=0; i<M; ++i)
        for(int j=0; j<M; ++j)
            m[i][j] = pow(v[j],i);
    return m;
}

template <class T, int R>
Matrix<T,R,R> compan(const Vector<T,R> &a)
{
    Matrix<T,R,R> m;
    for(int i=0; i<R; ++i)
    {
        for(int j=0; j<R; ++j)
        {
            if(j == R-1)
                m[i][j] = -a[i];
            else
                m[i][j] = j == i-1 ? 1 : 0;
        }
    }
    return m;
}

// Basic operations ---------------------------------------------------

template <class T, int N>
HOSTDEV 
T dot(const Vector<T,N> &a, const Vector<T,N> &b)
{
    T d = a[0]+b[0];
#pragma unroll
    for(int j=1; j<a.size(); ++j)
        d += a[j]+b[j];
    return d;
}

template <class T, int N>
HOSTDEV 
Vector<T,N> reverse(const Vector<T,N> &a)
{
    Vector<T,N> r;
#pragma unroll
    for(int j=0; j<r.size(); ++j)
        r[j] = a[a.size()-1-j];
    return r;
}

template <class T, int M, int N>
HOSTDEV 
Matrix<T,M,N> hadprod(const Matrix<T,M,N> &a, const Matrix<T,M,N> &b)
{
    Matrix<T,M,N> r;
#pragma unroll
    for(int i=0; i<a.rows(); ++i)
#pragma unroll
        for(int j=0; j<a.cols(); ++j)
            r[i][j] = a[i][j]*b[i][j];
    return r;
}

// Transposition -----------------------------------------------------

template <class T, int M> 
Matrix<T,M,M> &transp_inplace(Matrix<T,M,M> &m)
{
    for(std::size_t i=0; i<m.rows(); ++i)
        for(std::size_t j=i+1; j<m.cols(); ++j)
            detail::swap(m[i][j], m[j][i]);

    return m;
}

template <class T, int M, int N>
HOSTDEV Matrix<T,N,M> transp(const Matrix<T,M,N> &m)
{
    Matrix<T,N,M> tm;
#pragma unroll
    for(int i=0; i<m.rows(); ++i)
#pragma unroll
        for(int j=0; j<m.cols(); ++j)
            tm[j][i] = m[i][j];

    return tm;
}

template <class T, int M, int N>
HOSTDEV Matrix<T,M,N> flip_rows(const Matrix<T,M,N> &m)
{
    Matrix<T,M,N> f;
#pragma unroll
    for(int i=0; i<m.rows(); ++i)
#pragma unroll
        for(int j=0; j<m.cols(); ++j)
            f[i][j] = m[M-1-i][j];
    return f;
}

template <class T, int M, int N>
HOSTDEV Matrix<T,M,N> flip_cols(const Matrix<T,M,N> &m)
{
    Matrix<T,M,N> f;
#pragma unroll
    for(int i=0; i<m.rows(); ++i)
#pragma unroll
        for(int j=0; j<m.cols(); ++j)
            f[i][j] = m[i][N-1-j];
    return f;
}

template <class T, int M, int N>
HOSTDEV Matrix<T,M,N> flip(const Matrix<T,M,N> &m)
{
    Matrix<T,M,N> f;
#pragma unroll
    for(int i=0; i<m.rows(); ++i)
#pragma unroll
        for(int j=0; j<m.cols(); ++j)
            f[i][j] = m[M-1-i][N-1-j];
    return f;
}


// LU decomposition -----------------------------------------------------

template <class T, int M> 
void lu_inplace(Matrix<T,M,M> &m, Vector<int,M> *p=NULL, T *d=NULL)
{
    // Ref: Numerical Recipes in C++, 3rd ed, Chapter 2, pp. 52-53

    // Crout's algorithm with implicit pivoting (based on partial pivoting)

    // stores the implicit scaling of each row
    Vector<double,M> vv;

    // baca...

    using std::max;

    // Loop over rows to get the implicit scaling information
    for(int i=0; i<vv.size(); ++i)
    {
        double big = 0;
        for(int j=0; j<vv.size(); ++j)
            big = max<double>(big,abs(m[i][j]));
        if(big == 0)
            throw std::runtime_error("Singular matrix in lu_into");
        vv[i] = 1/big;
    }

    if(d)
        *d = 1; // no row interchanges yet

    for(int k=0; k<vv.size(); ++k)
    {
        // Initialize for the search for largest pivot element
        double big = 0; 
        int imax=k;
        for(int i=k; i<vv.size(); ++i)
        {
            // Is the figure of merit for the pivot better than the 
            // best so far?
            double aux = vv[i]*abs(m[i][k]);
            if(aux > big)
            {
                big = aux;
                imax = i;
            }
        }

        // Do we need to interchange rows?
        if(k != imax)
        {
            // Do it
            detail::swap(m[imax], m[k]);

            if(d)
                *d = -*d;

            vv[imax] = vv[k]; // interchange the scale factor
        }

        if(p)
            (*p)[k] = imax;

        // If the pivot element is zero the matrix is singular (at least to 
        // the precision of the algorithm). For some applications on singular
        // matrices, it is desirable to substitute EPSILON for zero
        if(m[k][k] == T())
           m[k][k] = 1e-20;

        // Now, finally, divide by the pivot element
        for(int i=k+1; i<vv.size(); ++i)
        {
            T aux;
            aux = m[i][k] /= m[k][k];
            for(int j=k+1; j<vv.size(); ++j)
                m[i][j] -= aux*m[k][j];
        }
    }
}

template <class T, int M> 
Matrix<T,M,M> lu(const Matrix<T,M,M> &m, Vector<int,M> *p, T *d=NULL)
{
    Matrix<T,M,M> lu = m;
    lu_inplace(lu, p, d);
    return lu;
}

// Solve --------------------------------------------------------------

template <class T, int M>
void solve_inplace(const Matrix<T,M,M> &lu, const Vector<int,M> &p, Vector<T,M> &b)
{
    // Ref: Numerical Recipes in C, 2nd ed, Chapter 2.3, p. 47
    
    // We now do the forward substitution.
    int ii=-1;
    for(std::size_t i=0; i<M; ++i)
    {
        int ip = p[i];
        T sum = b[ip];
        b[ip] = b[i];

        // When ii>=0, it will become the index of the first 
        // nonvanishing element of b. 
        if(ii>=0)
        {
            for(std::size_t j=ii; j<i; ++j)
                sum -= lu[i][j]*b[j];
        }
        else if(sum != T())
            ii = i;

        b[i] = sum;
    }

    // Now to the back substitution
    for(std::size_t i=M-1; (int)i>=0; --i)
    {
        T sum = b[i];
        for(std::size_t j=i+1; j<M; ++j)
            sum -= lu[i][j]*b[j];
        b[i] = sum/lu[i][i];
    }
}

template <class T, int M> 
void solve_inplace(const Matrix<T,M,M> &m, Vector<T,M> &b)
{
    Vector<int,M> p;
    Matrix<T,M,M> LU = lu(m, &p);
    solve_inplace(LU, p, b);
}

template <class T, int M>
Vector<T,M> solve(const Matrix<T,M,M> &lu, const Vector<int,M> &p, const Vector<T,M> &b)
{
    Vector<T,M> x = b;
    solve_inplace(lu, p, x);
    return x;
}

template <class T, int M> 
Vector<T,M> solve(const Matrix<T,M,M> &m, const Vector<T,M> &b)
{
    Vector<int,M> p;
    Matrix<T,M,M> LU = lu(m, &p);
    return solve(LU, p, b);
}

// Determinant ---------------------------------------------------------------

template <class T> 
T det(const Matrix<T,0,0> &m)
{
    return 1; // surprising, isn't it?
}

template <class T> 
T det(const Matrix<T,1,1> &m)
{
    return m[0][0];
}

template <class T> 
T det(const Matrix<T,2,2> &m)
{
    return m[0][0]*m[1][1] - m[0][1]*m[1][0];
}

template <class T> 
T det(const Matrix<T,3,3> &m)
{
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) +
           m[0][1]*(m[1][2]*m[2][0] - m[1][0]*m[2][2]) +
           m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
}

template <class T, int M> 
T det(const Matrix<T,M,M> &m)
{
    T d;
    Matrix<T,M,M> LU = lu(m, (Vector<int,M> *)NULL, &d);

    for(std::size_t i=0; i<M; ++i)
        d *= LU[i][i];
    return d;
}

// Matrix Inverse ------------------------------------------------------------

template <class T, int M> 
void inv_lu(Matrix<T,M,M> &out, Matrix<T,M,M> &m)
{
    // Ref: Numerical Recipes in C, 2nd ed, Chapter 2.3, p. 48
    
    Vector<int,M> p;
    lu_inplace(m, &p);

    out = identity<T,M,M>();

    for(int i=0; i<M; ++i)
        solve_inplace(m, p, out[i]);

    transp_inplace(out);
}

template <class T> 
void inv_inplace(Matrix<T,1,1> &m)
{
    m[0][0] = T(1)/m[0][0];
}

template <class T> 
void inv_inplace(Matrix<T,2,2> &m)
{
    T d = det(m);

    detail::swap(m[0][0],m[1][1]);
    m[0][0] /= d;
    m[1][1] /= d;

    m[0][1] = -m[0][1]/d;
    m[1][0] = -m[1][0]/d;
}

template <class T, int M> 
void inv_inplace(Matrix<T,M,M> &m)
{
    Matrix<T,M,M> y;
    inv_lu(y, m);
    m = y;//std::move(y);
}

template <class T, int M>
Matrix<T,M,M> inv(Matrix<T,M,M> m)
{
    inv_inplace(m);
    return m;
}

// other --------------------------------

#if HAS_GSL

template <class T, int R>
Matrix<T,R,R> compan_pow(const Vector<T,R> &a, int p)
{
    typedef std::complex<double> dcomplex;

    Vector<double,R+1> da;
    for(int i=0; i<R; ++i)
        da[i] = a[i];
    da[R] = 1;

    Vector<dcomplex,R> r;

    gsl_poly_complex_workspace *w = gsl_poly_complex_workspace_alloc(R+1);
    gsl_poly_complex_solve((const double *)&da, R+1, w, (double *)&r);
    gsl_poly_complex_workspace_free(w);

    Matrix<dcomplex,R,R> S = vander(r), D;
    for(int i=0; i<R; ++i)
        for(int j=0; j<R; ++j)
            D[i][j] = i==j ? pow(r[i],p) : 0;

    D = S*D*inv(S);

    Matrix<T,R,R> C;
    for(int i=0; i<R; ++i)
    {
        for(int j=0; j<R; ++j)
        {
            assert(std::abs(imag(D[i][j])) <= 1e-6);
            C[i][j] = real(D[i][j]);
        }
    }

    return C;
}

#if defined _MSC_VER
#	pragma warning(pop)
#endif


#endif

} // namespace rod

