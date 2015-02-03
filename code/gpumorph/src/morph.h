#ifndef GPUMORPH_MORPH_H
#define GPUMORPH_MORPH_H

#include <util/dimage.h>
#include "parameters.h"

class Pyramid;
struct PyramidLevel;

typedef bool (*ProgressCallback)(const std::string &text,
                                 int current, int maximum,
                                 const rod::dimage<float2> *halfway,
                                 const rod::dimage<float> *ssim_error,
                                 void *data);

class Morph
{
public:
    Morph(const Parameters &params);
    ~Morph();

	const Parameters &params() const { return m_params; };
	Parameters &params() { return m_params; };

    const rod::dimage<float3> &image0() const { return m_dimg0; }
    const rod::dimage<float3> &image1() const { return m_dimg1; }

    void set_callback(ProgressCallback cb, void *cbdata=NULL);

    bool calculate_halfway_parametrization(rod::dimage<float2> &out) const;

private:
    Parameters m_params;
    ProgressCallback m_cb;
    rod::dimage<float3> m_dimg0, m_dimg1;
    void *m_cbdata;

    Pyramid *m_pyramid;

    void cpu_optimize_level(PyramidLevel &lvl) const;

    void initialize_level(PyramidLevel &lvl) const;
    bool optimize_level(int &curiter, int maxiter, int totaliter,
        PyramidLevel &lvl, int orig_width, int orig_height, int nlevel) const;
};


void downsample(rod::dimage<float> &dest, const rod::dimage<float> &orig);
void upsample(PyramidLevel &dest, PyramidLevel &orig);


void render_halfway_image(rod::dimage<float3> &out, PyramidLevel &lvl,
                          const rod::dimage<float3> &in0,
                          const rod::dimage<float3> &in1);

void render_halfway_image(rod::dimage<float3> &out,
                          const rod::dimage<float2> &hwpar,
                          const rod::dimage<float3> &in0,
                          const rod::dimage<float3> &in1);

#endif
