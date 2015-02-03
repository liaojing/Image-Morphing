#ifndef IMAGE_H
#define IMAGE_H

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "color.h"
//opencv
#include <opencv2/opencv.hpp>
using namespace cv;

namespace image {
    template <class T = float> class rgba {
    public:
        rgba(void) {
            r = g = b = a = NULL;
            m_w = m_h = 0;
        }
        ~rgba() {
            free(r); free(g); free(b); free(a);
            r = g = b = a = NULL;
            m_w = m_h = 0;
        }
        int resize(int h, int w) {
            // try to allocate new buffers
            free(r); free(g); free(b); free(a);
            r = static_cast<T *>(calloc(w*h,sizeof(T)));
            g = static_cast<T *>(calloc(w*h,sizeof(T)));
            b = static_cast<T *>(calloc(w*h,sizeof(T)));
            a = static_cast<T *>(malloc(w*h*sizeof(T)));
            if (!r || !g || !b || !a) {
                free(r); free(g); free(b); free(a);
                r = g = b = a = NULL;
                w = h = 0;
                return 0;
            }
            // fill alpha with 1.0
            for (int i = 0; i < w*h; i++) a[i] = 1.f;
            m_h = h;
            m_w = w;
            return 1;
        }
        int load(const rgba<float> &other) {
            if (!resize(other.height(), other.width())) 
                return 0;
            memcpy(r, other.r, m_w*m_h*sizeof(T));
            memcpy(g, other.g, m_w*m_h*sizeof(T));
            memcpy(b, other.b, m_w*m_h*sizeof(T));
            memcpy(a, other.a, m_w*m_h*sizeof(T));
            return 1;
        }
        T *r, *g, *b, *a;
        void size(int *h, int *w) const { *h = m_h; *w = m_w; }
        int width(void) const { return m_w; }
        int height(void) const { return m_h; }
        void lrgb2srgb(void) {
            for (int i = 0; i < m_w*m_h; i++) 
                color::lrgb2srgb(r[i], g[i], b[i], r+i, g+i, b+i);
        }

        void normalize(void) {
            float min = FLT_MAX, max = -FLT_MAX; 
            for (int i = 0; i < m_w*m_h; i++)  {
                float r = this->r[i]; 
                float g = this->g[i]; 
                float b = this->b[i]; 
                if (r < min) min = r;
                if (g < min) min = g;
                if (b < min) min = b;
                if (r > max) max = r;
                if (g > max) max = g;
                if (b > max) max = b;
            }
            if (max > min) {
                float inv = 1.f/(max-min);
                for (int i = 0; i < m_w*m_h; i++)  {
                    this->r[i] -= min;
                    this->r[i] *= inv;
                    this->g[i] -= min;
                    this->g[i] *= inv;
                    this->b[i] -= min;
                    this->b[i] *= inv;
                }
            }
        }

        void srgb2lrgb(void) {
            for (int i = 0; i < m_w*m_h; i++) 
                color::srgb2lrgb(r[i], g[i], b[i], r+i, g+i, b+i);
        }
        void lrgb2lab(void) {
            for (int i = 0; i < m_w*m_h; i++) 
                color::lrgb2lab(r[i], g[i], b[i], r+i, g+i, b+i);
        }
        void lab2lrgb(void) {
            for (int i = 0; i < m_w*m_h; i++) 
                color::lab2lrgb(r[i], g[i], b[i], r+i, g+i, b+i);
        }
    private:
        int m_w, m_h;
        // prevent accidental shallow copy
        rgba(const rgba<float> &other);
        rgba<float>& operator=(const rgba<float> &other); 
    };

    // load and store
    template <class T> int load(cv::Mat &image, image::rgba<T> *rgba);
	template <class T> int store(cv::Mat &image, const image::rgba<T> &rgba);
}

#endif // IMAGE_H
