#include <util/image_ops.h>
#include <util/linalg.h>
#include <util/symbol.h>
#include <util/timer.h>
#include <util/dmath.h>
#include <sstream>
#include <assert.h>
#include "parameters.h"
#include "pyramid.h"
#include "stencils.h"
#include "config.h"
#include "morph.h"
#include "imgio.h"
#if CUDA_SM < 20
#   include <util/cuPrintf.cu>
#else
extern "C"
{
extern _CRTIMP __host__ __device__ __device_builtin__ int     __cdecl printf(const char*, ...);
}
#endif

__constant__ rod::Matrix<fmat5,5,5> c_tps_data;
__constant__ rod::Matrix<imat3,5,5> c_improvmask;
__constant__ imat3 c_improvmask_offset;
__constant__ rod::Matrix<imat5,5,5> c_iomask;

texture<float, 2, cudaReadModeElementType> tex_img0, tex_img1;

struct KernParameters/*{{{*/
{
    KernParameters() {}
    KernParameters(const Parameters &p)
        : w_ui(p.w_ui)
        , w_tps(p.w_tps)
        , w_ssim(p.w_ssim)
        , ssim_clamp(p.ssim_clamp)
        , eps(p.eps)
        , bcond(p.bcond)
    {
    }

    float w_ui, w_tps, w_ssim;
    float ssim_clamp;
    float eps;
    BoundaryCondition bcond;
};/*}}}*/

__device__ int isignbit(int i)/*{{{*/
{
    return (unsigned)i >> 31;
}/*}}}*/
__device__ int2 calc_border(int2 p, int2 dim)/*{{{*/
{
    int2 B;

    // I'm proud of the following lines, and they're faster too

#if 1
    int s = isignbit(p.x-2);
    int aux = p.x - (dim.x-2);
    B.x = p.x*s + (!s)*(2 + (!isignbit(aux))*(1+aux));

    s = isignbit(p.y-2);
    aux = p.y - (dim.y-2);
    B.y = p.y*s + (!s)*(2 + (!isignbit(aux))*(1+aux));
#endif


#if 0
    if(p.y==0)
        B.y = 0;
    else if(p.y==1)
        B.y = 1;
    else if(p.y==dim.y-2)
        B.y = 3;
    else if(p.y==dim.y-1)
        B.y = 4;
    else
        B.y = 2;

    if(p.x==0)
        B.x = 0;
    else if(p.x==1)
        B.x = 1;
    else if(p.x==dim.x-2)
        B.x = 3;
    else if(p.x==dim.x-1)
        B.x = 4;
    else
        B.x = 2;
#endif

    return B;
}/*}}}*/

// Auxiliary functions --------------------------------------------------------

__device__ float ssim(float2 mean, float2 var, float cross, /*{{{*/
                      float counter, float ssim_clamp)
{
    if(counter <= 0)
        return 0;

    const float c2 = pow2(255 * 0.03); // 58.5225

    mean /= counter;

#if 0
    var = (var-counter*mean*mean)/counter;
    var.x = max(0.0f, var.x);
    var.y = max(0.0f, var.y);
#endif

    var.x = fdimf(var.x, counter*mean.x*mean.x)/counter;
    var.y = fdimf(var.y, counter*mean.y*mean.y)/counter;

    cross = (cross - counter*mean.x*mean.y)/counter;
    /*

    float c3 = c2/2; // 29.26125f;
    float2 sqrtvar = sqrt(var);

    float c = (2*sqrtvar.x*sqrtvar.y + c2) / (var.x + var.y + c2),
          s = (abs(cross) + c3)/(sqrtvar.x*sqrtvar.y + c3);

    float value = c*s;

    */

    float value = (2*cross + c2)/(var.x+var.y+c2);

    return max(min(1.0f,value),ssim_clamp);
    //return saturate(1.0f-c*s);
}/*}}}*/

// Level processing -----------------------------------------------------------

#include "init.cu"

#include "optim.cu"

Morph::Morph(const Parameters &params)
    : m_cb(NULL)
    , m_cbdata(NULL)
    , m_params(params)
{
    load(m_dimg0, m_params.fname0);
    load(m_dimg1, m_params.fname1);

    if(m_dimg0.width()!=m_dimg1.width() || m_dimg0.height()!=m_dimg1.height())
        throw std::runtime_error("Images must have equal dimensions");

    m_pyramid = new Pyramid();
    try
    {
        create_pyramid(*m_pyramid, m_dimg0, m_dimg1,
                       params.start_res, params.verbose);
    }
    catch(...)
    {
        delete m_pyramid;
        throw;
    }
}

Morph::~Morph()
{
    delete m_pyramid;
}

void Morph::set_callback(ProgressCallback cb, void *cbdata)
{
    m_cb = cb;
    m_cbdata = cbdata;
}


bool Morph::calculate_halfway_parametrization(rod::dimage<float2> &out) const
{
    cpu_optimize_level(m_pyramid->back());

    rod::base_timer *morph_timer = NULL;
    if(m_params.verbose)
        morph_timer = &rod::timers.gpu_add("Morph",m_dimg0.width()*m_dimg0.height(),"P");

    int totaliter = 0;
    for(int i=1; i<m_pyramid->size(); ++i)
        totaliter += std::ceil((float)m_params.max_iter/pow(m_params.max_iter_drop_factor,i));

    int curiter = 0;
    int max_iter = m_params.max_iter;

    for(int l=m_pyramid->size()-2; l >= 0; --l)
    {
        max_iter = (int)((float)max_iter / m_params.max_iter_drop_factor);

        if(m_params.verbose)
            std::cout << "Processing level " << l << std::endl;

        rod::base_timer *timer = NULL;
        if(m_params.verbose)
        {
            std::ostringstream ss;
            ss << "Level " << l;

            timer = &rod::timers.gpu_add(ss.str(),(*m_pyramid)[l].width*(*m_pyramid)[l].height,"P");
        }

        upsample((*m_pyramid)[l], (*m_pyramid)[l+1]);
        initialize_level((*m_pyramid)[l]);
        if(!optimize_level(curiter, max_iter, totaliter, (*m_pyramid)[l],
                           (*m_pyramid)[0].width, (*m_pyramid)[0].height,l))
        {
            if(timer)
                timer->stop();
            if(morph_timer)
                morph_timer->stop();
            return false;
        }

        if(timer)
            timer->stop();
    }

    if(morph_timer)
        morph_timer->stop();

    internal_vector_to_image(out, (*m_pyramid)[0].v, (*m_pyramid)[0],
                             make_float2(1,1));

    return true;
}
