#ifndef GPUFILTER_MITCHELL_NETRAVALI_H
#define GPUFILTER_MITCHELL_NETRAVALI_H

namespace rod
{

struct mitchell_netravali_weights
{
    template <class T>
    __device__ void operator()(T alpha, T &w0, T &w1, T &w2, T &w3, int k=0)
    {
        T alpha2 = alpha*alpha;
        T one_alpha = 1-alpha,
          two_alpha = 2-alpha;

        switch(k)
        {
        case 0:
            w0 = 1/18.f+(-1/2.f+(5/6.f-7/18.f*alpha)*alpha)*alpha;
            w1 = 8/9.f+(-2+7/6.f*alpha)*alpha2;
            w2 = 8/9.f+(one_alpha*one_alpha)*(-15-21*alpha)/18.f;
            w3 = 16/9.f+two_alpha*(-60+(22+7*alpha)*two_alpha)/18.f;
            break;
        case 1:
            w0 = -1/2.f+(5/3.f-7/6.f*alpha)*alpha;
            w1 = (-4+7/2.f*alpha)*alpha;
            w2 = -(one_alpha*(-15-21*alpha))/9.f - 7*one_alpha*one_alpha/6.f;
            w3 = 10/3.f+two_alpha*(-30-21*alpha)/18.f;
            break;
        case 2:
            w0 = 5/3.f-7/3.f*alpha;
            w1 = -4+7*alpha;
            w2 = 3-7*alpha;
            w3 = -2/3.f+7/3.f*alpha;
            break;
        }
    }
};

inline float mitchell_netravali(float r)
{
    r = std::abs(r);

    if (r < 1.f) 
        return (16+r*r*(-36+21*r))/18.0f;
    else if (r < 2.f) 
        return (32+r*(-60+(36-7*r)*r))/18.0f;
    else 
        return 0.f;
}

}

#endif
