#ifndef DISCRETE_H
#define DISCRETE_H

#include <cstdlib>
#include <cstdio>
#include <complex>
#include "error.h"
#include "generating.h"

namespace kernel {

    // discrete kernels
    namespace discrete {

        class base {
        public:
            virtual ~base() { ; }
            virtual int support(void) const = 0;
            virtual const char *name(void) const = 0; 
            virtual base *clone(void) const  = 0; 
            const float *v;
        };

        template <class T>
        class cloner_base: public base{
        public:
            virtual base *clone(void) const {
                return new T(static_cast<const T &>(*this));
            }
        };

        static inline
        int isdelta(const float *kernel, int W) {
            return (W == 1 && kernel[0] == 1.f);
        }

        static inline 
        int isdelta(const base *kernel) {
            return isdelta(kernel->v, kernel->support());
        }

        class delta: public cloner_base<delta> {
        public:
            delta(void) { v = p_v; }
            const char *name(void) const { return "delta"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[1];
        };

        class sampled: public cloner_base<sampled> {
        public:
            sampled(::kernel::generating::base *ker) {
                m_W = ker->support();
                if (ker->support() % 2 == 0) m_W--;
                float *tmp_v = static_cast<float *>(malloc(m_W*sizeof(float)));
                if (!tmp_v) errorf(("out of memory!"));
                int r = m_W/2;
                for (int i = 0; i < m_W; i++)
                    tmp_v[i] = ker->value(r-i);
                v = tmp_v;
                sprintf(m_name, "[%.30s]", ker->name());
                delete ker;
            }
            virtual ~sampled(void) {
                free(const_cast<float*>(v));
            }
            int support(void) const { return m_W; }
            const char *name(void) const { return m_name; }
        private:
            int m_W;
            char m_name[256];
        };

        // This is the quasi-interpolator from "Quantitative Fourier 
        // Analysis of Approximation Techniques", Thierry Blu and 
        // Michael Unser, 1999
        class fir_blu35: public cloner_base<fir_blu35> {
        public:
            fir_blu35(void) { v = p_v; }
            const char *name(void) const { return "fir-blu35"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[9];
        };

        class ifir_blu35: public cloner_base<ifir_blu35> {
        public:
            ifir_blu35(void) { v = p_v; }
            const char *name(void) const { return "ifir-blu35"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[11];
        };

        // "Efficient Digital Pre-filtering for Least-Squares Linear 
        // Approximation", Marco Dalai, Riccardo Leonardi, and Pierangelo
        // Migliorati, 2006
        class fir_dalai1: public cloner_base<fir_dalai1> {
        public:
            fir_dalai1(void) { v = p_v; }
            const char *name(void) const { return "fir-dalai1"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class fir_dalai2: public cloner_base<fir_dalai2> {
        public:
            fir_dalai2(void) { v = p_v; }
            const char *name(void) const { return "fir-dalai2"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class fir_dalai3: public cloner_base<fir_dalai3> {
        public:
            fir_dalai3(void) { v = p_v; }
            const char *name(void) const { return "fir-dalai3"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        // "Beyond Interpolation", Laurent Condat, Thierry Blu, 
        // and Michael Unser, 2005
        class ifir_condat0: public cloner_base<ifir_condat0> {
        public:
            ifir_condat0(void) { v = p_v; }
            const char *name(void) const { return "ifir-condat0"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[3];
        };

        class ifir_condat1: public cloner_base<ifir_condat1> {
        public:
            ifir_condat1(void) { v = p_v; }
            const char *name(void) const { return "ifir-condat1"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[3];
        };

        class ifir_condat2: public cloner_base<ifir_condat2> {
        public:
            ifir_condat2(void) { v = p_v; }
            const char *name(void) const { return "ifir-condat2"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class ifir_condat3: public cloner_base<ifir_condat3> {
        public:
            ifir_condat3(void) { v = p_v; }
            const char *name(void) const { return "ifir-condat3"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class ifir_condat_omoms3: public cloner_base<ifir_condat_omoms3> {
        public:
            ifir_condat_omoms3(void) { v = p_v; }
            const char *name(void) const { return "ifir-condat-omoms3"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        // Some sampled autocorrelations used to produce duals 
        class a_hat: public cloner_base<a_hat> {
        public:
            a_hat(void) { v = p_v; }
            const char *name(void) const { return "[a_hat]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[3];
        };

        class a_bspline2: public cloner_base<a_bspline2> {
        public:
            a_bspline2(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline2]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_bspline3: public cloner_base<a_bspline3> {
        public:
            a_bspline3(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline3]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        class a_bspline4: public cloner_base<a_bspline4> {
        public:
            a_bspline4(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline4]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[9];
        };

        class a_bspline5: public cloner_base<a_bspline5> {
        public:
            a_bspline5(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline5]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[11];
        };

        class a_omoms2: public cloner_base<a_omoms2> {
        public:
            a_omoms2(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms2]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_omoms3: public cloner_base<a_omoms3> {
        public:
            a_omoms3(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms3]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        class a_omoms4: public cloner_base<a_omoms4> {
        public:
            a_omoms4(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms4]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[9];
        };

        class a_omoms5: public cloner_base<a_omoms5> {
        public:
            a_omoms5(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms5]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[11];
        };

        class a_mitchell_netravali: public cloner_base<a_mitchell_netravali> {
        public:
            a_mitchell_netravali(void) { v = p_v; }
            const char *name(void) const {return "[a_mitchell_netravali]";}
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        class a_keys4: public cloner_base<a_keys4> {
        public:
            a_keys4(void) { v = p_v; }
            const char *name(void) const { return "[a_keys4]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        class a_crt: public cloner_base<a_crt> {
        public:
            a_crt(void) { v = p_v; }
            const char *name(void) const { return "[a_crt]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[11];
        };

        // Some sampled crosscorrelations for consistent sampling
        class a_bspline3_box: public cloner_base<a_bspline3_box> {
        public:
            a_bspline3_box(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline3_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_omoms3_box: public cloner_base<a_omoms3_box> {
        public:
            a_omoms3_box(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms3_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_keys4_box: public cloner_base<a_keys4_box> {
        public:
            a_keys4_box(void) { v = p_v; }
            const char *name(void) const { return "[a_keys4_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_mitchell_netravali_box: public cloner_base<a_mitchell_netravali_box> {
        public:
            a_mitchell_netravali_box(void) { v = p_v; }
            const char *name(void) const { 
                return "[a_mitchell_netravali_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        class a_bspline5_box: public cloner_base<a_bspline5_box> {
        public:
            a_bspline5_box(void) { v = p_v; }
            const char *name(void) const { return "[a_bspline5_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        class a_omoms5_box: public cloner_base<a_omoms5_box> {
        public:
            a_omoms5_box(void) { v = p_v; }
            const char *name(void) const { return "[a_omoms5_box]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[7];
        };

        // Filter used by "Analytic Antialiasing with Prism Splines", 
        // 1995, McCool, to perform orthogonal projection into the space 
        // of cubic B-splines. Same as [bspline3] * [bspline3], which is 
        // of course wrong.
        class mccool: public cloner_base<mccool> {
        public:
            mccool(void) { v = p_v; }
            const char *name(void) const { return "[mccool]"; }
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            static const float p_v[5];
        };

        // "Recursive Gaussian derivative filters". 
        // Van Vliet, L. J., Young, I. T., and Verbeek, P. W. 1998. 
        class van_vliet: public cloner_base<van_vliet> {
        public:
            van_vliet(float s) { 
                sprintf(m_name, "[van-vliet(%g)]", s);
                double b10, a11;
                weights1(s, &b10, &a11);
                double b20, a21, a22;
                weights2(s, &b20, &a21, &a22);
                double inv_s = 1./(p2(b10)*p2(b20));
                p_v[0] = p_v[6] = static_cast<float>(inv_s*(a11*a22));
                p_v[1] = p_v[5] = static_cast<float>(inv_s*(a22 + p2(a11)*a22 
                    + a11*a21*(1 + a22)));
                p_v[2] = p_v[4] = static_cast<float>(inv_s*(a11*a22 + (1 
                    + p2(a11))*a21*(1 + a22) + a11*(1 + p2(a21) + p2(a22))));
                p_v[3] = static_cast<float>(inv_s*(2*a11*a21*(1 + a22) + (1 
                    + p2(a11)) * (1 +p2(a21) +p2(a22))));
                v = p_v;
            }
            const char *name(void) const { return m_name; } 
            int support(void) const { return sizeof(p_v)/sizeof(p_v[0]); }
        private:
            float p_v[7];
            char m_name[256];

            double p2(float t) const { return t*t; }

            double qs(double s) const { return .00399341 + .4715161*s; }

            std::complex<double> ds(std::complex<double> d, double s) const {
                double q = qs(s);
                return std::polar(pow(abs(d),1./q), arg(d)/q);
            }

            double ds(double d, double s) const { return pow(d, 1./qs(s)); }

            void weights1(double s, double *b0, double *a1) const {
                static const double d3(1.86543);
                double d = ds(d3, s);
                *b0 = (-(1.-d)/d);
                *a1 = (-1./d);
            }

            void weights2(double s, double *b0, double *a1, double *a2) const {
                static const std::complex<double> d1(1.41650, 1.00829);
                std::complex<double> d = ds(d1, s);
                double n2 = abs(d);
                n2 *= n2;
                double re = real(d);
                *b0 = ((1.-2.*re+n2)/n2);
                *a1 = (-2.*re/n2);
                *a2 = (1./n2);
            }
        };

    } // namespace discrete

} // namespace kernel

#endif
