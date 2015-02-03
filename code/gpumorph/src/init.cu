const int INIT_BW = 32,
          INIT_BH = 4,
          INIT_NB = 4;
__global__
__launch_bounds__(INIT_BW*INIT_BH, INIT_NB)
__global__ void kernel_initialize_level(KernPyramidLevel lvl,
                                        float ssim_clamp)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;

    int2 B = calc_border(pos,lvl.pixdim);

    int counter=0;
    float2 mean = {0,0}, 
           var = {0,0};
    float cross = 0;
    float2 tps_b = {0,0};

#pragma unroll
    for(int i=0; i<5; ++i)
    {
#pragma unroll
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j] == 0)
                continue;

            int nbidx = mem_index(lvl,pos + make_int2(j,i)-2);

            float2 v = lvl.v[nbidx];

            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || (v.x==0 && v.y==0));

            float2 tpos = make_float2((pos + make_int2(j,i) - 2)) + 0.5f;

            float2 luma;
            luma.x = tex2D(tex_img0, tpos.x - v.x, tpos.y - v.y),
            luma.y = tex2D(tex_img1, tpos.x + v.x, tpos.y + v.y);

            //assert(lvl.contains(pix.pos.x+j-2,pix.pos.y+i-2) || (luma.x==0 && luma.y==0));

            // this is the right thing to do, but result is better without it
            // luma *= c_iomask[B.y][B.x][i][j];

            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || c_tps_data[B.y][B.x][i][j]==0);
            tps_b += v*c_tps_data[B.y][B.x][i][j];


            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || c_iomask[B.y][B.x][i][j]==0);
            counter += c_iomask[B.y][B.x][i][j];
            mean += luma;
            var += luma*luma;
            cross += luma.x*luma.y;

            if(i==2 && j==2)
                lvl.ssim.luma[nbidx] = luma;
        }
    }

    int idx = mem_index(lvl, pos);
    lvl.ssim.counter[idx] = counter;
    lvl.ssim.mean[idx] = mean;
    lvl.ssim.var[idx] = var;
    lvl.ssim.cross[idx] = cross;
    lvl.ssim.value[idx] = ssim(mean, var, cross, counter, ssim_clamp);

    lvl.tps.axy[idx] = c_tps_data[B.y][B.x][2][2]/2;
    lvl.tps.b[idx] = tps_b;
}

__global__
void init_improving_mask(unsigned int *impmask, int bw, int bh)
{
    int bx = blockIdx.x*blockDim.x + threadIdx.x,
        by = blockIdx.y*blockDim.y + threadIdx.y;

    if(bx >= bw || by >= bh)
        return;

    if(bx==0 || by==0 || bx == bw-1 || by == bh-1)
        impmask[by*bw+bx] = 0;
    else
        impmask[by*bw+bx] = (1<<25)-1;

}

void Morph::initialize_level(PyramidLevel &lvl) const
{
    rod::Matrix<fmat5,5,5> tps;
    calc_tps_stencil(tps);
    copy_to_symbol(c_tps_data,tps);

    rod::Matrix<imat3,5,5> improvmask_check;
    rod::Matrix<int,3,3> improvmask_off;
    calc_nb_improvmask_check_stencil(lvl, improvmask_check, improvmask_off);
    copy_to_symbol(c_improvmask,improvmask_check);
    copy_to_symbol(c_improvmask_offset,improvmask_off);

    rod::Matrix<imat5,5,5> iomask;
    calc_nb_io_stencil(lvl, iomask);
    copy_to_symbol(c_iomask,iomask);

    lvl.ssim.mean.fill(0);
    lvl.ssim.var.fill(0);
    lvl.ssim.luma.fill(0);
    lvl.ssim.cross.fill(0);
    lvl.ssim.value.fill(0);
    lvl.ssim.counter.fill(0);

    lvl.tps.axy.fill(0);
    lvl.tps.b.fill(0);

    lvl.ui.axy.fill(0);
    lvl.ui.b.fill(0);

    tex_img0.normalized = false;
    tex_img0.filterMode = cudaFilterModeLinear;
    tex_img0.addressMode[0] = tex_img0.addressMode[1] = cudaAddressModeClamp;

    tex_img1.normalized = false;
    tex_img1.filterMode = cudaFilterModeLinear;
    tex_img1.addressMode[0] = tex_img1.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex_img0, lvl.img0);
    cudaBindTextureToArray(tex_img1, lvl.img1);

    dim3 bdim(INIT_BW,INIT_BH),
         gdim((lvl.width+bdim.x-1)/bdim.x,
              (lvl.height+bdim.y-1)/bdim.y);

    rod::base_timer *timer  = NULL;
    if(m_params.verbose)
        timer = &rod::timers.gpu_add("init",lvl.width*lvl.height,"P");

    kernel_initialize_level<<<gdim,bdim>>>(lvl, m_params.ssim_clamp);

    // initialize ui data in cpu since usually there aren't so much points

    std::vector<float> ui_axy(lvl.ui.axy.size(), 0);
    std::vector<float2> ui_b(lvl.ui.b.size(), make_float2(0,0)),
                        v;

    lvl.v.copy_to_host(v);

    for(size_t i=0; i<m_params.ui_points.size(); ++i)
    {
        const ConstraintPoint &cpt = m_params.ui_points[i];

        float2 p0 = make_float2(cpt.lp*make_float2((float)lvl.width, (float)lvl.height)-0.5f),
               p1 = make_float2(cpt.rp*make_float2((float)lvl.width, (float)lvl.height)-0.5f);
		
	p0.x = max(0.f,p0.x);
	p0.y = max(0.f,p0.y);
	p1.x = max(0.f,p1.x);
	p1.y = max(0.f,p1.y);

        float2 con = (p0+p1)/2,
               pv = (p1-p0)/2;

        for(int y=(int)floor(con.y); y<=(int)ceil(con.y); ++y)
        {
            if(y >= lvl.height)
                break;
            for(int x=(int)floor(con.x); x<=(int)ceil(con.x); ++x)
            {
                if(x >= lvl.width)
                    break;

                int idx = mem_index(lvl, make_int2(x,y));

                using std::abs;

                float bilinear_w = (1 - abs(y-con.y))*(1 - abs(x-con.x));
                ui_axy[idx] += bilinear_w;
                ui_b[idx] += 2*bilinear_w*(v[idx] - pv);
            }
        }
    }

    lvl.ui.axy.copy_from_host(ui_axy);
    lvl.ui.b.copy_from_host(ui_b);

    int2 blk = make_int2((lvl.width+4)/5+2, (lvl.height+4)/5+2);

    gdim = dim3((blk.x + bdim.x-1)/bdim.x,
                (blk.y + bdim.y-1)/bdim.y);

    init_improving_mask<<<gdim,bdim>>>(lvl.improving_mask, blk.x, blk.y);

    if(timer)
        timer->stop();
}

