#include <cfloat>

__constant__ KernParameters c_params;

const int OPT_BW = 32,
          OPT_BH = 8,
          OPT_NB = 5;

const int SPACING = 5;

template <int BW, int BH>
struct SSIMData
{
    float2 mean[BH*2+4][BW*2+4],
           var[BH*2+4][BW*2+4];
    float  cross[BH*2+4][BW*2+4],
           value[BH*2+4][BW*2+4];

    int2 orig;
};

template <class T>
__device__ void swap(T &a, T &b)/*{{{*/
{
    T temp = a;
    a = b;
    b = temp;
}/*}}}*/

// returns -1 if pixel cannot improve due to neighbors (and itself) 
// not improving
__device__ int get_improve_mask_idx(const KernPyramidLevel &lvl, /*{{{*/
                            const int2 &p)
{
    int2 block = p/5;
    int2 offset = p%5;

    int begi = (offset.y >= 2 ? 1 : 0),
        begj = (offset.x >= 2 ? 1 : 0),
        endi = begi+2,
        endj = begj+2;

    int impmask_idx = (block.y+1)*lvl.impmask_rowstride + (block.x+1);

    for(int i=begi; i<endi; ++i)
    {
        for(int j=begj; j<endj; ++j)
        {
            int d = impmask_idx + c_improvmask_offset[i][j];

            if(lvl.improving_mask[d]&c_improvmask[offset.y][offset.x][i][j])
                return impmask_idx;
        }
    }

    return -1;
}/*}}}*/

__device__ bool pixel_on_border(const KernPyramidLevel &lvl, const int2 &p)/*{{{*/
{
    switch(c_params.bcond)
    {
    case BCOND_NONE:
        break;
    case BCOND_CORNER:
        if(p.x==0 && p.y==0 || p.x==0 && p.y==lvl.pixdim.y-1 ||
           p.x==lvl.pixdim.x-1 && p.y==0 && p.x==lvl.pixdim.x-1 && p.y==lvl.pixdim.y-1)
        {
            return true;
        }
        break;
    case BCOND_BORDER:
        if(p.x==0 || p.y==0 || p.x==lvl.pixdim.x-1 || p.y==lvl.pixdim.y-1)
            return true;
        break;
    }
    return false;
}/*}}}*/

// gradient calculation --------------------------

template <int BW, int BH>
__device__ float ssim_change(const KernPyramidLevel &lvl,/*{{{*/
                            const int2 &p,
                            float2 v, float2 old_luma, 
                            const SSIMData<BW,BH> &ssimdata)
{
    float2 luma;

    luma.x = tex2D(tex_img0, p.x-v.x + 0.5f, p.y-v.y + 0.5f),
    luma.y = tex2D(tex_img1, p.x+v.x + 0.5f, p.y+v.y + 0.5f);

    float change = 0;

    float2 dmean = luma - old_luma,
           dvar = pow2(luma) - pow2(old_luma);
    float  dcross = luma.x*luma.y - old_luma.x*old_luma.y;

    bool need_counter = p.x < 4 || p.x >= lvl.pixdim.x-4 ||
                        p.y < 4 || p.y >= lvl.pixdim.y-4;

    int idx = mem_index(lvl, p);
    int2 B = calc_border(p, lvl.pixdim);

    for(int i=0; i<5; ++i)
    {
        int sy = p.y+i-2 - ssimdata.orig.y;
        assert(sy >= 0 && sy < OPT_BH*2+4);
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j] == 0)
                continue;

            int sx = p.x+j-2 - ssimdata.orig.x;

            int nb = mem_index(lvl, p + make_int2(j,i)-2);

            float2 mean, var;
            float counter = need_counter ? lvl.ssim.counter[nb] : 25,
                  cross;

            assert(sx >= 0 && sx < OPT_BW*2+4);

            mean = ssimdata.mean[sy][sx];
            var = ssimdata.var[sy][sx];
            cross = ssimdata.cross[sy][sx];

            mean += dmean;
            var +=  dvar;
            cross += dcross;

            float new_ssim = ssim(mean,var,cross,counter,c_params.ssim_clamp);
            change += ssimdata.value[sy][sx] - new_ssim;
        }
    }

    return change;
}/*}}}*/

template <int BW, int BH>
__device__ float energy_change(const KernPyramidLevel &lvl, /*{{{*/
                               const int2 &p,
                               const float2 &v,
                               const float2 &old_luma,
                               const float2 &d,
                               const SSIMData<BW,BH> &ssimdata)
{
    float v_ssim = ssim_change(lvl, p, v+d, old_luma, ssimdata);

    int idx = mem_index(lvl,p);

    float v_tps = lvl.tps.axy[idx]*(d.x*d.x + d.y*d.y);
    v_tps += lvl.tps.b[idx].x*d.x;
    v_tps += lvl.tps.b[idx].y*d.y;

    float v_ui  = lvl.ui.axy[idx]*(d.x*d.x + d.y*d.y);
    v_ui += lvl.ui.b[idx].x*d.x;
    v_ui += lvl.ui.b[idx].y*d.y;

    return (c_params.w_ui*v_ui + c_params.w_ssim*v_ssim)*lvl.inv_wh
                + c_params.w_tps*v_tps;
}/*}}}*/

template <int BW, int BH>
__device__ float2 compute_gradient(const KernPyramidLevel &lvl, /*{{{*/
                                   const int2 &p,
                                   const float2 &v,
                                   const float2 &old_luma,
                                   const SSIMData<BW,BH> &ssimdata)
{
    float2 g;
    g.x = energy_change(lvl,p,v,old_luma,make_float2(c_params.eps,0),ssimdata)-
          energy_change(lvl,p,v,old_luma,make_float2(-c_params.eps,0),ssimdata);
    g.y = energy_change(lvl,p,v,old_luma,make_float2(0,c_params.eps),ssimdata)-
          energy_change(lvl,p,v,old_luma,make_float2(0,-c_params.eps),ssimdata);
    return -g;
}/*}}}*/

// foldover --------------------------------

template <int X, int Y, int SIGN>
__device__ float2 fover_calc_vtx(const KernPyramidLevel &lvl,/*{{{*/
                                 const int2 &p, float2 v)
{
    const int2 off = make_int2(X,Y);

    if(lvl.contains(p+off))
        v = SIGN*lvl.v[mem_index(lvl,p+off)];

     return v + (p-off);
}/*}}}*/

__device__ void fover_update_isec_min(float2 c, float2 grad,/*{{{*/
                                      float2 e0, float2 e1,
                                      float &t_min)
{
    float2 de = e1-e0,
           dce = c-e0;

    // determinant
    float d  = de.y*grad.x - de.x*grad.y;

    // signals that we don't have an intersection (yet)
    // t = td/d
    float td = -1;

    // u = ud/d
    // e0 + u*(e1-e0) = intersection point
    float ud = grad.x*dce.y - grad.y*dce.x;

    int sign = signbit(d);

    // this is faster than multiplying ud and d by sign
    if(sign)
    {
        ud = -ud;
        d = -d;
    }

    // line by c0 and c1 intersects segment [e0,e1] ?
    if(ud >= 0 && ud <= d) // u >= 0 && u <= 1
    {
        // c0 + t*(c1-c0) = intersection point
        td = de.x*dce.y - de.y*dce.x;
        td *= (-sign*2+1);

        if(td >= 0 && td < t_min*d)
            t_min = td/d;
    }
}/*}}}*/

template <int SIGN>
__device__ void fover_calc_isec_min(const KernPyramidLevel &lvl, /*{{{*/
                                    const int2 &p,
                                    float2 v, float2 grad, 
                                    float &t_min)
{
    // edge segment, start from upper left (-1,-1), go cw around center
    // pixel testing whether pixel will intersect the edge or not
    float2 e[2] = { fover_calc_vtx<-1,-1,SIGN>(lvl, p, v),
                    fover_calc_vtx< 0,-1,SIGN>(lvl, p, v)};

    float2 efirst = e[0];

    // pixel displacement (c0 -> c1)
    float2 c = p + v;

    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<1,-1,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<1,0,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<1,1,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<0,1,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<-1,1,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<-1,0,SIGN>(lvl, p, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    fover_update_isec_min(c,grad,e[1],efirst,t_min);
}/*}}}*/

__device__ float prevent_foldover(const KernPyramidLevel &lvl,/*{{{*/
                                  const int2 &p, 
                                  float2 v, float2 grad)
{
    float t_min = 10;

    fover_calc_isec_min<-1>(lvl, p, -v, -grad, t_min);
    fover_calc_isec_min<1>(lvl, p, v, grad, t_min);

    return max(t_min-c_params.eps,0.0f);
}/*}}}*/

template <int BW, int BH>
__device__ void golden_section_search(const KernPyramidLevel &lvl,/*{{{*/
                                      const int2 &p,
                                      float a, float c,
                                      float2 v, float2 grad,
                                      float2 old_luma,
                                      const SSIMData<BW,BH> &ssimdata,
                                      float &fmin, float &tmin)
{
    const float R = 0.618033989f,
                C = 1.0f - R;


    float b = a*R + c*C,  // b between [a,c>
          x = b*R + c*C;  // x between [b,c>

    float fb = energy_change(lvl, p, v, old_luma, grad*b, ssimdata),
          fx = energy_change(lvl, p, v, old_luma, grad*x, ssimdata);

#pragma unroll 4
    while(c - a > c_params.eps)
    {
        if(fx < fb) // bracket is [b,x,c] ?
        {
            // [a,b,c] = [b,x,c]
            a = b;
            b = x;
            x = b*R + c*C; // x between [b,c>
        }
        else // bracket is [a,b,x] ?
        {
            // [a,b,c] = [a,b,x]
            c = x;
            x = b*R + a*C; // x between <a,b]
        }

        float f = energy_change(lvl, p, v, old_luma, grad*x, ssimdata);

        if(fx < fb)
        {
            fb = fx;
            fx = f;
        }
        else
        {
            swap(b,x);
            fx = fb;
            fb = f;
        }
    }

    if(fx < fb)
    {
        tmin = x;
        fmin = fx;
    }
    else
    {
        tmin = b;
        fmin = fb;
    }
}/*}}}*/

// update --------------------------------

template <int BW, int BH>
__device__ void ssim_update(KernPyramidLevel &lvl,/*{{{*/
                            const int2 &p, 
                            float2 v, float2 old_luma,
                            SSIMData<BW,BH> &ssimdata)
{
    float2 luma;

    luma.x = tex2D(tex_img0, p.x-v.x + 0.5f, p.y-v.y + 0.5f),
    luma.y = tex2D(tex_img1, p.x+v.x + 0.5f, p.y+v.y + 0.5f);

    int idx = mem_index(lvl,p);

    lvl.ssim.luma[idx] = luma;

    float2 dmean = luma - old_luma,
           dvar = pow2(luma) - pow2(old_luma);
    float  dcross = luma.x*luma.y - old_luma.x*old_luma.y;

    int2 B = calc_border(p, lvl.pixdim);

    for(int i=0; i<5; ++i)
    {
        int sy = p.y+i-2 - ssimdata.orig.y;
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j])
            {
                int sx = p.x+j-2 - ssimdata.orig.x;

                atomicAdd(&ssimdata.mean[sy][sx], dmean);
                atomicAdd(&ssimdata.var[sy][sx], dvar);
                atomicAdd(&ssimdata.cross[sy][sx], dcross);
            }
        }
    }
}/*}}}*/

template <int BW, int BH>
__device__ void commit_pixel_motion(KernPyramidLevel &lvl, /*{{{*/
                                    const int2 &p,
                                    const float2 &newv,
                                    const float2 &old_luma,
                                    const float2 &grad,
                                    SSIMData<BW,BH> &ssimdata)

{
    ssim_update(lvl, p, newv, old_luma, ssimdata);

    int2 B = calc_border(p, lvl.pixdim);

    // tps update
    for(int i=0; i<5; ++i)
    {
        for(int j=0; j<5; ++j)
        {
            assert(lvl.contains(p.x+j-2,p.y+i-2) || c_tps_data[B.y][B.x][i][j] == 0);

            int nb = mem_index(lvl, p + make_int2(j,i)-2);
            atomicAdd(&lvl.tps.b[nb], grad*c_tps_data[B.y][B.x][i][j]);
        }
    }

    int idx = mem_index(lvl,p);

    // ui update
    lvl.ui.b[idx] += 2*grad*lvl.ui.axy[idx];

    // vector update
    lvl.v[idx] = newv;
}/*}}}*/

// optimization kernel --------------------------

template <int BW, int BH>
__device__ bool optimize_pixel(const KernPyramidLevel &lvl,/*{{{*/
                               const int2 &p,
                               const SSIMData<BW,BH> &ssim,
                               float2 &old_luma,
                               float2 &v,
                               float2 &grad,
                               int &impmask_idx)
{
    if(lvl.contains(p))
    {
        int idx = mem_index(lvl,p);

        v = lvl.v[idx],
        old_luma = lvl.ssim.luma[idx];

        impmask_idx = get_improve_mask_idx(lvl, p);

        assert(lvl.contains(p) || lvl.improving_mask[impmask_idx] == 0);

        if(impmask_idx >= 0)
        {
            if(!pixel_on_border(lvl, p))
            {
                grad = compute_gradient(lvl, p, v, old_luma, ssim);

            //    float ng = hypot(grad.x,grad.y); // slower
                float ng = sqrt(pow2(grad.x)+pow2(grad.y));

                if(ng != 0)
                {
                    grad /= ng;

                    float t = prevent_foldover(lvl, p, v, grad);

                    float tmin, fmin;

                    golden_section_search(lvl, p, 0, t,
                                          v, grad, old_luma, ssim, fmin, tmin);

                    if(fmin < 0)
                    {
                        grad *= tmin;
                        v += grad;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}/*}}}*/

template <template<int,int> class F>
__device__ void process_shared_state(F<8,8> fun, const KernPyramidLevel &lvl,/*{{{*/
                                     const int2 &block_orig)
{
    const int BW = 8, BH = 8;

    /*     BW      BW      4
       -----------------------
       |        |        |   | BH
       |   1    |   2    | 6 |
       |-----------------|---|
       |        |        |   | BH
       |   4    |   3    | 6 |
       |-----------------|---|
       |   5    |   5    | 7 | 4
       -----------------------
    */

    // area 1
    int sx = threadIdx.x,
        sy = threadIdx.y;
    int2 pix = block_orig + make_int2(sx,sy);
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 2
    pix.x += BW;
    sx += BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 3
    pix.y += BH;
    sy += BH;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 4
    pix.x -= BW;
    sx -= BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 5
    sx = (threadIdx.y/4)*BW + threadIdx.x;
    sy = threadIdx.y%4 + BH*2;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 6
    sx = threadIdx.x%4 + BW*2;
    sy = threadIdx.y*(BW/4) + threadIdx.x/4;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 7
    sy += BH*2;
    pix.y += BH*2;
    if(lvl.contains(pix) && sy < BH*2+4)
        fun(pix, sx, sy);
}/*}}}*/

template <template<int,int> class F>
__device__ void process_shared_state(F<32,8> fun, const KernPyramidLevel &lvl,/*{{{*/
                                     const int2 &block_orig)
{
    const int BW = 32, BH = 8;

    int sx = threadIdx.x,
        sy = threadIdx.y;

    /*     BW      BW      4
       -----------------------
       |        |        |   | BH
       |   1    |   2    | 6 |
       |-----------------|---|
       |        |        |   | BH
       |   4    |   3    | 6 |
       |-----------------|---|
       |   5    |   5    | 6 | 4
       -----------------------
    */

    // area 1
    int2 pix = block_orig + make_int2(sx,sy);
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 2
    pix.x += BW;
    sx += BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 3

    pix.y += BH;
    sy += BH;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 4

    pix.x -= BW;
    sx -= BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 5
    sx = (threadIdx.y/4)*BW + threadIdx.x;
    sy = threadIdx.y%4 + BH*2;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix) && sx < BW*2+4 && sy < BH*2+4)
        fun(pix, sx, sy);

    // area 6
    sx = threadIdx.x%4 + BW*2;
    sy = threadIdx.y*8 + threadIdx.x/4;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix) && sx < BW*2+4 && sy < BH*2+4)
        fun(pix, sx, sy);
}/*}}}*/

template <int BW, int BH>
class LoadSSIM/*{{{*/
{
public:
    __device__ LoadSSIM(const KernPyramidLevel &lvl, SSIMData<BW,BH> &ssim)
        : m_level(lvl), m_ssim(ssim) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix);
        m_ssim.mean[sy][sx] = m_level.ssim.mean[idx];
        m_ssim.var[sy][sx] = m_level.ssim.var[idx];
        m_ssim.cross[sy][sx] = m_level.ssim.cross[idx];
        m_ssim.value[sy][sx] = m_level.ssim.value[idx];
    }

private:
    const KernPyramidLevel &m_level;
    SSIMData<BW,BH> &m_ssim;
};/*}}}*/

template <int BW, int BH>
class SaveSSIM/*{{{*/
{
public:
    __device__ SaveSSIM(KernPyramidLevel &lvl, const SSIMData<BW,BH> &ssim)
        : m_level(lvl), m_ssim(ssim) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix);
        m_level.ssim.mean[idx] = m_ssim.mean[sy][sx];
        m_level.ssim.var[idx] = m_ssim.var[sy][sx];
        m_level.ssim.cross[idx] = m_ssim.cross[sy][sx];
        m_level.ssim.value[idx] = m_ssim.value[sy][sx];
    }

private:
    KernPyramidLevel &m_level;
    const SSIMData<BW,BH> &m_ssim;
};/*}}}*/

template <int BW, int BH>
class UpdateSSIM/*{{{*/
{
public:
    __device__ UpdateSSIM(const KernPyramidLevel &lvl, SSIMData<BW,BH> &ssim)
        : m_level(lvl), m_ssim(ssim) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix);
        m_ssim.value[sy][sx] = ssim(m_ssim.mean[sy][sx],
                                    m_ssim.var[sy][sx],
                                    m_ssim.cross[sy][sx],
                                    m_level.ssim.counter[idx],
                                    c_params.ssim_clamp);
    }

private:
    const KernPyramidLevel &m_level;
    SSIMData<BW,BH> &m_ssim;
};/*}}}*/

__global__
//__launch_bounds__(OPT_BW*OPT_BH, OPT_NB)
void kernel_optimize_level(KernPyramidLevel lvl,/*{{{*/
                           int offx, int offy,
                           bool *out_improving)
{

    __shared__ SSIMData<OPT_BW,OPT_BH> ssim;

    {
        int2 block_orig = make_int2(blockIdx.x*(OPT_BW*2+SPACING)+offx-2,
                                    blockIdx.y*(OPT_BH*2+SPACING)+offy-2);

        if(threadIdx.x == 0 && threadIdx.y == 0)
            ssim.orig = block_orig;

        process_shared_state(LoadSSIM<OPT_BW,OPT_BH>(lvl, ssim), lvl, block_orig);
    }

    bool improving = false;

    __syncthreads();

    for(int i=0; i<2; ++i)
    {
        for(int j=0; j<2; ++j)
        {
            int2 p = ssim.orig + make_int2(threadIdx.x*2+j+2,
                                           threadIdx.y*2+i+2);

            float2 old_luma, v, grad;
            int impmask_idx = -1;
            bool ok = optimize_pixel(lvl, p, ssim, old_luma, v, grad, 
                                     impmask_idx);

            int2 offset = p%5;
            __syncthreads();

            if(ok)
            {
                commit_pixel_motion(lvl, p, v, old_luma, grad, ssim);

                improving = true;
                atomicOr(&lvl.improving_mask[impmask_idx], 
                         1 << (offset.x + offset.y*5));
            }
            else if(impmask_idx >= 0)
            {
                atomicAnd(&lvl.improving_mask[impmask_idx], 
                          ~(1 << (offset.x + offset.y*5)));
            }
            __syncthreads();

            process_shared_state(UpdateSSIM<OPT_BW,OPT_BH>(lvl, ssim), lvl, ssim.orig);

            __syncthreads();
        }
    }

    process_shared_state(SaveSSIM<OPT_BW,OPT_BH>(lvl, ssim), lvl, ssim.orig);

    if(improving)
        *out_improving = true;
}/*}}}*/

template <class T>
T *addressof(T &v)
{
    return reinterpret_cast<T*>(&const_cast<char &>(reinterpret_cast<const volatile char &>(v)));
}

bool Morph::optimize_level(int &curiter, int maxiter, int totaliter, 
                           PyramidLevel &lvl, int orig_width, int orig_height, 
                           int nlevel) const
{
    dim3 bdim(OPT_BW,OPT_BH),
         gdim((lvl.width+OPT_BW*2+SPACING-1)/(OPT_BW*2+SPACING),
              (lvl.height+OPT_BH*2+SPACING-1)/(OPT_BH*2+SPACING));

    rod::base_timer *timer = NULL;
    if(m_params.verbose)
        timer = &rod::timers.gpu_add("optimize",lvl.width*lvl.height,"P");

    KernPyramidLevel klvl(lvl);
    KernParameters kparams(m_params);

    rod::copy_to_symbol(c_params,kparams);

    bool *improving = NULL;
    cudaHostAlloc(&improving, sizeof(bool), cudaHostAllocMapped);
    rod::check_cuda_error("cudaHostAlloc");
    assert(improving != NULL);

    bool *dimproving = NULL;
    cudaHostGetDevicePointer(&dimproving, improving, 0);
    rod::check_cuda_error("cudaHostGetDevicePointer");

    int iter = 0;

    rod::cpu_timer cb_send_image_timer(0,"",false);

    try
    {
        do
        {
            if(m_cb != NULL)
            {
                std::ostringstream ss;
                ss << "Optimizing level " << nlevel <<  " iteration #" << iter+1;

                rod::dimage<float2> halfway;

                clock_t now = clock();

                if(cb_send_image_timer.is_stopped() || 
                   cb_send_image_timer.elapsed() >= 0.2)
                {
                    cb_send_image_timer.start();

                    internal_vector_to_image(halfway, lvl.v, lvl,
                                             make_float2((float)orig_width/lvl.width,
                                                         (float)orig_height/lvl.height));

                    if(orig_width != lvl.width || orig_height != lvl.height)
                    {
                        rod::dimage<float2> temp(orig_width, orig_height);
                        upsample(&temp, &halfway, rod::INTERP_LINEAR);

                        swap(halfway,temp); // move temp to halfway
                    }
                }

                if(!m_cb(ss.str(), ++curiter, totaliter, 
                       halfway.empty() ? NULL : addressof(halfway), NULL, 
                       m_cbdata))
                {
                    if(timer)
                        timer->stop();

                    return false;
                }
            }

            *improving = false;

            kernel_optimize_level<<<gdim,bdim>>>(klvl, 0,0, dimproving);
            kernel_optimize_level<<<gdim,bdim>>>(klvl, OPT_BW*2, 0, dimproving);
            kernel_optimize_level<<<gdim,bdim>>>(klvl, 0, OPT_BH*2, dimproving);
            kernel_optimize_level<<<gdim,bdim>>>(klvl, OPT_BW*2, OPT_BH*2, dimproving);

            cudaDeviceSynchronize();

            ++iter;
        }
        while(*improving && iter <= maxiter);

        curiter += maxiter-iter;

        cudaFreeHost(improving);
    }
    catch(...)
    {
        if(timer)
            timer->stop();

        cb_send_image_timer.stop();


        cudaFreeHost(improving);
        throw;
    }

    cb_send_image_timer.stop();

    if(timer)
        timer->stop();

    if(m_params.verbose)
        std::cout << iter << " iterations, grid " << gdim.x << 'x' << gdim.y << std::endl;

    return true;
}
