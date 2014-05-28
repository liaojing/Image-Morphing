#ifndef GPUFILTER_UTIL_H
#define GPUFILTER_UTIL_H

// most of these utilities are in C++TR1 or C++11,
// but were supporting C++03, so here they are...

#include <limits> // for numeric_limits
#include <string>

namespace rod
{

inline std::string strupr(const std::string &str)
{
    std::string ret;
    ret.reserve(str.size());

    for(size_t i=0; i<ret.size(); ++i)
        ret.push_back(toupper(str[i]));
    return ret;
}

template <bool EN, class T=void>
struct enable_if
{
    typedef T type;
};

template <class T>
struct enable_if<false,T>
{
};

template <class FROM, class TO>
struct is_convertible
{
private:
    struct yes {};
    struct no {yes m[2]; };

    static yes can_convert(TO *) {}
    static no can_convert(const void *) {}

public:
    static const bool value 
        = sizeof(can_convert((FROM *)NULL)) == sizeof(yes);
};

template <class T>
struct remove_const
{
    typedef T type;
};

template <class T>
struct remove_const<const T>
{
    typedef T type;
};

template <class T>
struct remove_volatile
{
    typedef T type;
};

template <class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};

template <class T>
struct remove_cv
{
    typedef typename remove_const<typename remove_volatile<T>::type>::type type;
};

template <class T, int N>
struct array
{
    // public so that it can be initialized with {...}
    T data[N];

    typedef T *iterator;
    typedef const T *const_iterator;

    __host__ __device__
    const T &operator[](int i) const { return data[i]; }

    __host__ __device__
    T &operator[](int i) { return data[i]; }

    size_t size() const { return N; }

    iterator begin() { return &data[0]; }
    iterator end() { return &data[N-1]; }

    const_iterator begin() const { return &data[0]; }
    const_iterator end() const { return &data[N-1]; }

    const_iterator cbegin() const { return &data[0]; }
    const_iterator cend() const { return &data[N-1]; }
};

// for mutable * -> const * conversion
template <class T, int N>
class array<const T *, N>
{
    const T *m_data[N];
public:
    typedef T **iterator;
    typedef const T **const_iterator;

    __host__ __device__
    array()
    {
    }

    __host__ __device__
    array(const array &that)
    {
        for(int i=0; i<N; ++i)
            m_data[i] = that[i];
    }

    __host__ __device__
    array(const array<T*,N> &that)
    {
        for(int i=0; i<N; ++i)
            m_data[i] = that[i];
    }

    __host__ __device__
    const T *&operator[](int i) { return m_data[i]; }

    __host__ __device__
    const T *const&operator[](int i) const { return m_data[i]; }

    size_t size() const { return N; }

    iterator begin() { return &m_data[0]; }
    iterator end() { return &m_data[N-1]; }

    const_iterator begin() const { return &m_data[0]; }
    const_iterator end() const { return &m_data[N-1]; }

    const_iterator cbegin() const { return &m_data[0]; }
    const_iterator cend() const { return &m_data[N-1]; }
};

template <class FROM, class TO>
struct copy_const
{
    typedef TO type;
};

template <class FROM, class TO>
struct copy_const<const FROM, TO>
{
    typedef const TO type;
};

template <class T>
struct is_const
{
    static const bool value = false;
};

template <class T>
struct is_const<const T>
{
    static const bool value = true;
};

template <class T>
struct is_volatile
{
    static const bool value = false;
};

template <class T>
struct is_volatile<volatile T>
{
    static const bool value = true;
};

template <class T>
struct is_integral 
{ 
    static const bool value 
        = std::numeric_limits<typename remove_cv<T>::type>::is_integer;
};

template <class T, class U>
struct is_same
{
    static const bool value = false;
};

template <class T>
struct is_same<T,T>
{
    static const bool value = true;
};

}


// from boost::preprocessor
// concatenate expanded arguments

#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) a ## b

#define PP_CAT3(a, b, c) PP_CAT(a,PP_CAT(b,c))

#endif
