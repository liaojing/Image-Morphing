#include <complex>
#include "image_ops.h"
#include "recfilter.h"

namespace rod
{

// gaussian blur ---------------------------------------------------------

namespace
{
    typedef std::complex<double> dcomplex;
    const dcomplex d1(1.41650, 1.00829);
    const double d3(1.86543);

    double qs(double s) {
        return .00399341 + .4715161*s;
    }

    double ds(double d, double s) {
        return pow(d, 1.0/qs(s));
    }

    dcomplex ds(dcomplex d, double s)
    {
        double q = qs(s);
        return std::polar(pow(abs(d),1.0/q), arg(d)/q);
    }

    void gaussian_weights1(double s, Vector<float,2> &w)
    {
        double d = ds(d3, s);

        int sign;
        if(rec_op(1,1)==0)
            sign = -1;
        else
            sign = 1;

        w[0] = static_cast<float>(-(1.0-d)/d);
        w[1] = sign*static_cast<float>(1.0/d);
    }

    void gaussian_weights2(double s, Vector<float,3> &w)
    {
        dcomplex d = ds(d1, s);
        double n2 = abs(d);
        n2 *= n2;
        double re = real(d);

        int sign;
        if(rec_op(1,1)==0)
            sign = -1;
        else
            sign = 1;

        w[0] = static_cast<float>((1-2*re+n2)/n2);
        w[1] = sign*static_cast<float>(2*re/n2);
        w[2] = sign*static_cast<float>(-1/n2);
    }

}

struct gaussian_blur_plan
{
    gaussian_blur_plan() : plan1(NULL), plan2(NULL) {}
    ~gaussian_blur_plan()
    {
        free(plan1);
        free(plan2);
    }

    recfilter5_plan *plan1, *plan2;
};

gaussian_blur_plan *gaussian_blur_create_plan(int width, int height,
                                              int rowstride, float sigma)
{
    gaussian_blur_plan *plan = new gaussian_blur_plan();
    try
    {
        update_plan(plan, width, height, rowstride, sigma);

        return plan;
    }
    catch(...)
    {
        delete plan;
        throw;
    }
}

void free(gaussian_blur_plan *plan)
{
    delete plan;
}

void update_plan(gaussian_blur_plan *plan, int width, int height, 
                 int rowstride, float sigma)
{
    // TODO: must have strong exception guarantee!

    Vector<float,1+1> weights1;
    gaussian_weights1(sigma,weights1);

    if(plan->plan1 == NULL)
    {
        plan->plan1 = recfilter5_create_plan<1>(width, height, 
                                                rowstride, weights1);
    }
    else
        update_plan<1>(plan->plan1, width, height, rowstride, weights1);

    Vector<float,1+2> weights2;
    gaussian_weights2(sigma,weights2);

    if(plan->plan2 == NULL)
    {
        plan->plan2 = recfilter5_create_plan<2>(width, height, 
                                                rowstride, weights2);
    }
    else
        update_plan<2>(plan->plan2, width, height, rowstride, weights2);

}

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out, 
                   dimage_ptr<const float,C> in)
{
    assert(plan != NULL);

    for(int i=0; i<C; ++i)
        recfilter5(plan->plan1, out[i], in[i]);

    for(int i=0; i<C; ++i)
        recfilter5(plan->plan2, out[i]);
}

template
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float> out, 
                   dimage_ptr<const float> in);

template
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float,3> out, 
                   dimage_ptr<const float,3> in);

} // namespace rod


