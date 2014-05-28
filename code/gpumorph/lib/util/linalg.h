#ifndef RECFILTER_LINALG_H
#define RECFILTER_LINALG_H

#include <vector>
#include <cassert>
#include "hostdev.h"
#include "linalg_fwd.h"

namespace rod
{

template <class T, int N>
class Vector 
{
public:
    std::vector<T> to_vector() const;

    HOSTDEV int size() const { return N; }

    HOSTDEV const T &operator[](int i) const;
    HOSTDEV T &operator[](int i);

    HOSTDEV operator const T *() const { return &m_data[0]; }
    HOSTDEV operator T *() { return &m_data[0]; }

    template <int R>
    HOSTDEV Vector<T,R> subv(int beg) const
    {
        assert(beg+R <= N);
        Vector<T,R> v;
        for(int i=beg; i<beg+R; ++i)
            v[i-beg] = m_data[i];
        return v;
    }

private:
    T m_data[N];
};


template <class T, int M, int N>
class Matrix
{
public:
    HOSTDEV int rows() const { return M; }
    HOSTDEV int cols() const { return N; }

    HOSTDEV const Vector<T,N> &operator[](int i) const;
    HOSTDEV Vector<T,N> &operator[](int i);

    HOSTDEV Vector<T,M> col(int j) const;
    HOSTDEV void set_col(int j, const Vector<T,M> &c);

private:
    Vector<T,N> m_data[M];
};

} // namespace rod

#include "linalg.hpp"

#endif
