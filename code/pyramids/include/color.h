#ifndef COLOR_H
#define COLOR_H

#include <cmath>

namespace color {

	// apply sRGB non-linearity to value
	static inline float srgbcurve(float f) {
		//if (f < 0.f) f = 0.f;
		//if (f > 1.f) f = 1.f;
        const float a = 0.055f;
        if (f <= 0.0031308f) return 12.92f*f;
        else return (1.f + a)*powf(f, 1.f/2.4f)-a;
		// return powf(f, 1/2.2f); // faster approximation
		// return sqrtf(f); // even faster 
		//return f; // disable 
	}

	// move from linear RGB to sRGB 
	static inline void lrgb2srgb(float lr, float lg, float lb, 
        float *sr, float *sg, float *sb) {
        *sr = srgbcurve(lr);
        *sg = srgbcurve(lg);
        *sb = srgbcurve(lb);
	}

	// remove sRGB non-linearity from value
	static inline float srgbuncurve(float f) {
		//if (f < 0.f) f = 0.f;
		//if (f > 1.f) f = 1.f;
        const float a = 0.055f;
        if (f <= 0.04045f) return f/12.92f;
        else return powf((f+a)/(1.f+a), 2.4f);
		// return powf(f, 2.2f); // faster approximation
		// return f*f; // even faster 
		//return f; // disable 
	}

	// move from sRGB to linear RGB
	static inline void srgb2lrgb(float sr, float sg, float sb, 
        float *lr, float *lg, float *lb) {
        *lr = srgbuncurve(sr);
        *lg = srgbuncurve(sg);
        *lb = srgbuncurve(sb);
	}

    // apply LAB non-linearity
    static inline float labcurve(float t) {
        if (t > 0.008856f) return powf(t, 1.f/3.f);
        else return (903.3f*t+16.f)/116.f;
    }
    
	// move from linear RGB to perceptual LAB
	static inline void lrgb2lab(float lr, float lg, float lb, 
        float *l, float *a, float *b) {
        float X = 0.412424f  * lr + 0.357579f * lg + 0.180464f  * lb;
        float Y = 0.212656f  * lr + 0.715158f * lg + 0.0721856f * lb;
        float Z = 0.0193324f * lr + 0.119193f * lg + 0.950444f * lb;
        X /= 0.95047f;
        Z /= 1.08883f;
        X = labcurve(X);
        Y = labcurve(Y);
        Z = labcurve(Z);
        *l = (116.f*Y)-16.f;
        *a = 500.f*(X-Y);
        *b = 200.f*(Y-Z);
	}

    // remove LAB non-linearity
    static inline float labuncurve(float t) {
        float pf = powf(t, 3.f);
        if (pf > 0.008856f) return pf; 
        else return (t-16.f/116.f)/7.787f;
    }

	// move from perceptual LAB to linear RGB
	static inline void lab2lrgb(float l, float a, float b, 
        float *lr, float *lg, float *lb) {
        float Y = (l+16.f)/116.f;
        float X = (a/500.f)+Y;
        float Z = Y-(b/200.f);
        X = labuncurve(X);
        Y = labuncurve(Y);
        Z = labuncurve(Z);
        X *= 0.95047f;
        Z *= 1.08883f;
        *lr = X * 3.24071f     + Y * (-1.53726f)  + Z * (-0.498571f);
        *lg = X * (-0.969258f) + Y * 1.87599f     + Z * 0.0415557f;
        *lb = X * 0.0556352f   + Y * (-0.203996f) + Z * 1.05707f;
    }
}

#endif
