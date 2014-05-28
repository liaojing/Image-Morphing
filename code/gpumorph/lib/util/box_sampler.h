#ifndef NLFILTER_BOX_SAMPLER_H
#define NLFILTER_BOX_SAMPLER_H

#include "dmath.h"

namespace rod
{

template <class S>
class box_sampler
{
public:
    typedef S sampler_type;
    typedef typename S::result_type result_type;

    template <class S2>
    struct rebind_sampler
    {
        typedef box_sampler<S2> type;
    };

    __device__ inline 
    result_type operator()(float2 pos, int kx=0, int ky=0) const
    {
        S sampler;

        if(kx == 0 && ky == 0)
            return sampler(pos.x, pos.y);
        else
        {
            // TODO: we must do something sensible here,
            // box derivatives are 0!
            return pixel_traits<result_type>::make_pixel(0);
        }
    }
};

}

#endif
