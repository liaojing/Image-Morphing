#ifndef NLFILTER_EFFECTS_H
#define NLFILTER_EFFECTS_H

#include "dmath.h"

// maps to/from yiq space from/to rgb space ------------------------------

namespace rod
{

__device__ inline
float rgb2yiq(float gray)
{
    return gray;
}

__device__ inline
float yiq2rgb(float y)
{
    return y;
}

__device__ inline
float3 rgb2yiq(float3 rgb)
{
    return make_float3(0.299f*rgb.x + 0.587f*rgb.y + 0.114f*rgb.z,
                       0.595f*rgb.x - 0.274453f*rgb.y - 0.321263f*rgb.z,
                       0.211456f*rgb.x - 0.522591f*rgb.y + 0.311135f*rgb.z);
}

__device__ inline
float3 yiq2rgb(float3 yiq)
{
    return make_float3(yiq.x + 0.9563f*yiq.y + 0.6210f*yiq.z,
                       yiq.x - 0.2721f*yiq.y - 0.6474f*yiq.z,
                       yiq.x - 1.1070f*yiq.y + 1.7046f*yiq.z);
}

// maps to/from gamma space from/to linear space ------------------------------

// apply sRGB non-linearity to value
__device__ inline 
float lrgb2srgb(float v)
{
    v = saturate(v);
//    const float a = 0.055f;
    if(v <= 0.0031308f)
        return 12.92f*v;
    else
        //return (1.f+a)*powf(v,1.f/2.4f)-a;
//        return powf(v,1/2.2f); // faster approximation
        return sqrtf(v); // even faster

}

__device__ inline 
float3 lrgb2srgb(float3 v)
{
    return make_float3(lrgb2srgb(v.x),lrgb2srgb(v.y),lrgb2srgb(v.z));
}

// remove sRGB non-linearity from value
__device__ inline 
float srgb2lrgb(float v)
{
    v = saturate(v);
//    const float a = 0.055f;
    if(v <= 0.04045f)
        return v/12.92f;
    else
 //       return powf((v+a)/(1.f+a),2.4f);
//        return powf(v,2.2f); // faster approximation
        return v*v; // even faster
}

__device__ inline 
float3 srgb2lrgb(float3 v)
{
    return make_float3(srgb2lrgb(v.x), srgb2lrgb(v.y), srgb2lrgb(v.z));
}


// posterize ------------------------------------------------------------

template <class T>
__device__ inline 
T posterize(T v, int levels)
{
    return saturate(rint(v*(levels-1))/(levels-1));
}

// threshold ------------------------------------------------------------

__device__ inline 
float threshold(float v, float amin, float amax)
{
    return v < amin || v > amax ? 0 : 1;
}

__device__ inline 
float3 threshold(float3 v, float amin, float amax)
{
    return make_float3(threshold(v.x, amin,amax), 
                       threshold(v.y, amin,amax), 
                       threshold(v.z, amin,amax));
}

// grayscale  ------------------------------------------------------------

__device__ inline 
float grayscale(float3 in)
{
    return 0.2126f * in.x + 0.7152f*in.y + 0.0722f*in.z;
}

// luminance --------------------------------------------------
//
__device__ inline 
float luminance(float in)
{
    return in;
}

__device__ inline 
float luminance(float3 in)
{
    // 'in' must be in linear space
    return  .299f * in.x + .587f * in.y + .114f * in.z;
}

// scale -------------------------------------------------------------

template <class T>
__device__ inline 
T scale(T v, float a)
{
    return v*a;
}

// bias -------------------------------------------------------------

template <class T>
__device__ inline 
T bias(T v, float a)
{
    return v+a;
}

// replacement -------------------------------------------------------------

__device__ inline 
float replacement(float v, float3 old, float3 new_, float3 tau)
{
    float a = fabs(v-grayscale(old));

    if(a <= tau.x && a <= tau.y && a <= tau.z)
        return saturate(grayscale(new_)+a);
    else
        return v;
}

__device__ inline 
float3 replacement(float3 v, float3 old, float3 new_, float3 tau)
{
    float3 a = fabs(v-old);

    if(a.x <= tau.x && a.y <= tau.y && a.z <= tau.z)
        return saturate(new_+a);
    else
        return v;
}

// polynomial ---------------------------------------------------------------

__device__ inline 
float3 polynomial(float3 v, int N, float *coeff)
{
    float3 cur = make_float3(0,0,0),
           power = make_float3(1,1,1);
#pragma unroll
    for(int i=0; i<N; ++i)
    {
        cur += coeff[i]*power;
        power *= v;
    }
    return saturate(cur);
}

// root --------------------------------------------------------------------

__device__ inline 
float root(float v, float n)
{
    if(v < 0 || v < 0 || v < 0)
        v = 0;

    return saturate(pow(v,1/n));
}

__device__ inline 
float3 root(float3 v, float n)
{
    if(v.x < 0 || v.y < 0 || v.z < 0)
        v = make_float3(0,0,0);

    return saturate(pow(v,1/n));
}

// gradient_edge_detection ------------------------------------------------

template <class T>
__device__ inline 
T gradient_edge_detection(T dx, T dy)
{
    return saturate(sqrt(dx*dx+dy*dy));
}

// laplacian  ----------------------------------------------------

template <class T>
__device__ inline 
T laplacian(T dxx, T dyy)
{
    return saturate(dxx+dyy);
}

// laplace_edge_enhacement -----------------------------------------


template <class T>
__device__ inline 
T laplace_edge_enhancement(T v, T dxx, T dyy, float multiple)
{
    return saturate(v - multiple*(dxx+dyy));
}

// yaroslavsky bilateral --------------------------------------------
namespace detail
{

__device__ inline
float calc_yb_g_tilde(float param, float E)
{
    if(param < 0.000001f)
        return 1.f/6;
    else
        return (param*exp(-(param*param)))/(3.f*E);
}


__device__ inline
float3 calc_yb_g_tilde(float3 param, float3 E)
{
    return make_float3(calc_yb_g_tilde(param.x,E.x),
                       calc_yb_g_tilde(param.y,E.y),
                       calc_yb_g_tilde(param.z,E.z));
}

__device__ inline
float calc_yb_f_tilde(float param, float g_tilde)
{
    if(param < 0.000001f)
        return 1.f/6;
    else
        return 3.f*g_tilde+((3.f*g_tilde-0.5f)/(param*param));
}

__device__ inline
float3 calc_yb_f_tilde(float3 param, float3 g_tilde)
{
    return make_float3(calc_yb_f_tilde(param.x,g_tilde.x),
                       calc_yb_f_tilde(param.y,g_tilde.y),
                       calc_yb_f_tilde(param.z,g_tilde.z));
}
}

template <class T>
__device__ inline
T yaroslavsky_bilateral(T v, T dx, T dy, T dxy, T dxx, T dyy,
                        float rho, float h)
{
    T grad = sqrt(dx*dx + dy*dy),
      ort = (1.f/(grad*grad))*(dx*dx*dxx + 2*dx*dy*dxy + dy*dy*dyy),
      tan = (1.f/(grad*grad))*(dy*dy*dxx - 2*dx*dy*dxy + dx*dx*dyy),
      param = grad*rho / h;

    const float sqrt_pi = 1.77245385;
    T E = 2*((sqrt_pi/2.f)*erff(param));

    T g_tilde = detail::calc_yb_g_tilde(param, E),
      f_tilde = detail::calc_yb_f_tilde(param, g_tilde);

    return saturate(v + rho*rho*(f_tilde*ort + g_tilde*tan));
}

// brightness and contrast ---------------------------------------------

template <class T>
__device__ inline
T brightness_contrast(T v, float brightness, float contrast)
{
    if(brightness < 0)
        v *= (1+brightness);
    else
        v += (1-v)*brightness;

    const float PI = 3.14159265359;

    float slant = tan((contrast+1)*PI/4);
    return saturate((v-0.5)*slant + 0.5);
}

// hue, saturation and lightness -----------------------------------

__device__ inline
float3 rgb2hsl(float3 rgb)
{
    float3 hsl;

    float M,m,C,h_prime;

    M = fmax(fmax(rgb.x, rgb.y), rgb.z);
    m = fmin(fmin(rgb.x, rgb.y), rgb.z);
    C = M-m;


    if (C == 0.f) 
        h_prime = 100.f; //too big value for h_prime means H is not defined!    

    if (M == rgb.x) 
        h_prime = fmod((rgb.y-rgb.z)/C,6.f);
    else if (M==rgb.y) 
        h_prime = ((rgb.z-rgb.x)/C)+2.f;
    else h_prime = ((rgb.x-rgb.y)/C)+4.f;

    hsl.x = 60.f*h_prime;
    hsl.z = 0.5f*(M+m);
    if (C==0) 
        hsl.y = 0.f;
    else 
        hsl.y = C/(1.f-fabs(2*(hsl.z)-1));

    return hsl;
}

__device__ inline
float3 hsl2rgb(float3 hsl)
{
    float C,h_prime,X,r1,g1,b1,m;

    C = (1-fabs(2*(hsl.z)-1.f))*(hsl.y);
    h_prime = (hsl.x)/60.f;
    X = C*(1-fabs(fmod(h_prime,2.f)-1.f));

    if (h_prime<1.f) {
        r1 = C;
        g1 = X;
        b1 = 0.f;
    }   
    else if (h_prime<2.f){
        r1 = X;
        g1 = C; 
        b1 = 0.f;
    }   
    else if (h_prime<3.f){
        r1 = 0.f;
        g1 = C;
        b1 = X; 
    }   
    else if (h_prime<4.f){
        r1 = 0.f;
        g1 = X; 
        b1 = C;
    }   
    else if (h_prime<5.f){
        r1 = X;
        g1 = 0.f;
        b1 = C;
    }   
    else if (h_prime<6.f) {
        r1 = C;
        g1 = 0.f;
        b1 = X;
    }
    else { // undefined value of H is mapped to (0,0,0)
        r1 = 0.f;
        g1 = 0.f;
        b1 = 0.f;
    } 

    m = hsl.z - C/2.f;

    float3 rgb;

    rgb.x = r1+m;
    rgb.y = g1+m;
    rgb.z = b1+m;

    return rgb;
}


__device__ inline
float3 hue_saturation_lightness(float3 v, float hue, float saturation,
                                float lightness)
{
    float3 hsl = rgb2hsl(v);

    if(hsl.x+hue < 0)
        hsl.x += 360 + hue;
    else if(hsl.x+hue > 360)
        hsl.x += hue-360;
    else
        hsl.x += hue;

    saturation /= 100;
    hsl.y = saturate(hsl.y*(1 + saturation));

    lightness = lightness/100 * 0.5;
    if(lightness < 0)
        hsl.z = saturate(hsl.z*(1 + lightness));
    else
        hsl.z = saturate(hsl.z + (1-hsl.z)*lightness);

    return hsl2rgb(hsl);
}


// this makes no sense whatsoever, but is included for completeness
__device__ inline
float hue_saturation_lightness(float v, float hue, float saturation,
                               float lightness)
{
    return grayscale(hue_saturation_lightness(make_float3(v,v,v),hue,saturation,lightness));
}

// unsharp mask -----------------------------------

__device__ inline
float3 unsharp_mask(float3 v, float blurred_vy, float amount, float threshold)
{
    // v must be in linear space
    float3 v_yiq = rgb2yiq(v);

    float dy = v_yiq.x - blurred_vy;

    v_yiq.x += v_yiq.x*amount*(dy < threshold ? dy*dy : dy);

    return yiq2rgb(v_yiq);
}

__device__ inline
float unsharp_mask(float v, float blurred_vy, float amount, float threshold)
{
    float dy = v - blurred_vy;

    return v+ v*amount*(dy < threshold ? dy*dy : dy);
}

} // namespace rod

#endif
