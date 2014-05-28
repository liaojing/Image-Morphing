#ifndef GPUMORPH_PYRAMID_H
#define GPUMORPH_PYRAMID_H

#include <vector>
#include <util/dvector.h>
#include <util/dimage_fwd.h>

struct PyramidLevel;

class Pyramid
{
public:
    ~Pyramid();

    PyramidLevel &append_new(int w, int h);

    PyramidLevel &operator[](int idx) { return *m_data[idx]; }
    const PyramidLevel &operator[](int idx) const { return *m_data[idx]; }

    PyramidLevel &back() { return *m_data.back(); }
    const PyramidLevel &back() const { return *m_data.back(); }

    size_t size() const { return m_data.size(); }

private:
    std::vector<PyramidLevel*> m_data;
};

struct PyramidLevel
{
    PyramidLevel(int w, int h);
    ~PyramidLevel();

    cudaArray *img0, *img1;

    int rowstride;// number of floats between two rows
    int impmask_rowstride;

    int width, height;
    float inv_wh;

    struct
    {
        rod::dvector<float2> mean, var, luma; // .x = img0, .y = img1
        rod::dvector<float> cross, value,
                            counter; // num of neighbors
    } ssim;

    struct
    {
        rod::dvector<float> axy;
        rod::dvector<float2> b;
    } tps, ui;

    rod::dvector<float2> v;

    rod::dvector<unsigned int> improving_mask;

private:
    // make it non-copyable
    PyramidLevel(const PyramidLevel &);
    PyramidLevel &operator=(const PyramidLevel &);
};

struct KernPyramidLevel
{
    KernPyramidLevel(PyramidLevel &lvl);

    struct
    {
        float *cross;
        float2 *var, *mean, *luma;
        float *value, *counter;
    } ssim;

    struct
    {
        float *axy;
        float2 *b;
    } tps, ui;

    float2 *v;

    unsigned int *improving_mask;

#if CUDA_SM >= 20
    template <class T>
    __device__
    bool contains_x(T x) const
        { return x >= 0 && x < pixdim.x; }
    template <class T>
    __device__ bool contains_y(T y) const
        { return y >= 0 && y < pixdim.y; }

    template <class T>
    __device__
    bool contains(T p) const // T == int2, float2, ...
        { return contains_x(p.x) && contains_y(p.y); }
    template <class T, class U>
    __device__
    bool contains(T x, U y) const // T == int2, float2, ...
        { return contains_x(x) && contains_y(y); }
#else
    __device__ bool contains_x(float x) const
        { return x >= 0 && x < pixdim.x; }
    __device__ bool contains_x(int x) const
        { return x >= 0 && x < pixdim.x; }
    __device__ bool contains_y(float y) const
        { return y >= 0 && y < pixdim.y; }
    __device__ bool contains_y(int y) const
        { return y >= 0 && y < pixdim.y; }
    __device__ bool contains(float2 p) const
        { return contains_x(p.x) && contains_y(p.y); }
    __device__ bool contains(int2 p) const
        { return contains_x(p.x) && contains_y(p.y); }
    __device__ bool contains(int x, int y) const
        { return contains_x(x) && contains_y(y); }
    __device__ bool contains(float x, float y) const
        { return contains_x(x) && contains_y(y); }
#endif

    int2 pixdim;
    int rowstride, impmask_rowstride;
    float inv_wh;
};

HOSTDEV inline int mem_index(int rowstride, int2 pos)
{
    return pos.y*rowstride + pos.x;
}

inline __device__ int mem_index(const KernPyramidLevel &lvl, int2 pos)
{
    return mem_index(lvl.rowstride, pos);
}

inline int mem_index(const PyramidLevel &lvl, int2 pos)
{
    return mem_index(lvl.rowstride, pos);
}

struct Parameters;

void create_pyramid(Pyramid &pyr,
                    const rod::dimage<float3> &img0,
                    const rod::dimage<float3> &img1,
                    int start_res, bool verbose=false);

template <class T>
void image_to_internal_vector(rod::dvector<T> &dest,
                              const rod::dimage<T> &orig,
                              const PyramidLevel &lvl,
                              T mult);

template <class T>
void internal_vector_to_image(rod::dimage<T> &dest,
                              const rod::dvector<T> &orig,
                              const PyramidLevel &lvl,
                              T mult);

#endif
