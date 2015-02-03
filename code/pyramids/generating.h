#ifndef GENERATING_H 
#define GENERATING_H

#include <cmath>

namespace kernel {

    namespace generating {

        class base {
        public:
            virtual ~base() { ; }
            virtual bool normalize(void) const = 0;
            virtual int support(void) const = 0;
            virtual bool primal(void) const = 0;
            virtual float value(float r) const = 0;
            virtual const char* name(void) const = 0;
            virtual base *new_d(void) const { return NULL; } 
            virtual base *new_dd(void) const { return NULL; }
            virtual base *clone(void) const = 0; 
        };

        template <class T>
        class cloner_base: public base {
            virtual base *clone(void) const {
                return new T(static_cast<const T &>(*this));
            }
        };

        namespace d {
            class bspline3: public ::kernel::generating::cloner_base<bspline3> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 4; }
                float value(float r) const {
                    if (r < -2.f) return 0.f; 
                    else if (r < -1.f) return (12.f+r*(12.f+3.f*r))/6.f;
                    else if (r < 0.f) return ((-12.f-9.f*r)*r)/6.f;
                    else if (r < 1.f) return (r*(-12.f+9.f*r))/6.f;
                    else if (r < 2.f) return  (-12.f+(12.f-3.f*r)*r)/6.f;
                    else return 0.f;
                }
                const char *name(void) const { return "d-bspline3"; }
            };

            class bspline5: public ::kernel::generating::cloner_base<bspline5> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 6; }
                float value(float r) const {
                    if (r < -3.f) return 0.f; 
                    else if (r < -2.f) return (405.f + r*(540.f 
                        + r*(270.f + r*(60.f + 5.f*r))))/120.f; 
                    else if (r < -1.f) return (-75.f + r*(-420.f 
                        + r*(-450.f + (-180.f - 25.f*r)*r)))/120.f; 
                    else if (r < 0.f) return (r*(-120.f 
                        + r*r*(120.f + 50.f*r)))/120.f; 
                    else if (r < 1.f) return (r*(-120.f 
                        + (120.f - 50.f*r)*r*r))/120.f; 
                    else if (r < 2.f) return (75.f + r*(-420.f 
                        + r*(450.f + r*(-180.f + 25.f*r))))/120.f; 
                    else if (r < 3.f) return (-405.f + r*(540.f 
                        + r*(-270.f + (60.f - 5.f*r)*r)))/120.f; 
                    else return 0.f;
                }
                const char *name(void) const { return "d-bspline5"; }
            };

            class bspline7: public ::kernel::generating::cloner_base<bspline7> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 8; }
                float value(float r) const {
                    if (r < -4.f) return 0.f;
                    else if (r < -3.f) return (28672 + r*(43008 + r*(26880 
                        + r*(8960 + r*(1680 + r*(168 + 7*r))))))/5040.f;
                    else if (r < -2.f) return (-12152 + r*(-38640 + r*(-41160 
                        + r*(-21280 + r*(-5880 + (-840 - 49*r)*r)))))/5040.f;
                    else if (r < -1.f) return (392 + r*(-1008 + r*(5880 
                        + r*(10080 + r*(5880 + r*(1512 + 147*r))))))/5040.f;
                    else if (r < 0.f) return (r*(-3360 + r*r*(2240 + (-840 
                        - 245*r)*r*r)))/5040.f;
                    else if (r < 1.f) return (r*(-3360 + r*r*(2240 + r*r*(-840 
                        + 245*r))))/5040.f;
                    else if (r < 2.f) return (-392 + r*(-1008 + r*(-5880 
                        + r*(10080 + r*(-5880 + (1512 - 147*r)*r)))))/5040.f;
                    else if (r < 3.f) return (12152 + r*(-38640 + r*(41160 
                        + r*(-21280 + r*(5880 + r*(-840 + 49*r))))))/5040.f;
                    else if (r < 4.f) return (-28672 + r*(43008 + r*(-26880 
                        + r*(8960 + r*(-1680 + (168 - 7*r)*r)))))/5040.f;
                    else return 0.f;
                }
                const char *name(void) const { return "d-bspline7"; }
            };
        }

        namespace dd {

            class bspline3: public ::kernel::generating::cloner_base<bspline3> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 4; }
                float value(float r) const {
                    if (r < -2.f) return 0.f; 
                    else if (r < -1.f) return (12.f+6.f*r)/6.f;
                    else if (r < 0.f) return (-12.f-18.f*r)/6.f;
                    else if (r < 1.f) return (-12.f+18.f*r)/6.f;
                    else if (r < 2.f) return  (12.f-6.f*r)/6.f;
                    else return 0.f;
                }
                const char *name(void) const { return "dd-bspline3"; }
            };


            class bspline5: public ::kernel::generating::cloner_base<bspline5> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 6; }
                float value(float r) const {
                    if (r < -3.f) return 0.f; 
                    else if (r < -2.f) return  (540.f + r*(540.f 
                        + r*(180.f + 20.f*r)))/120.f;
                    else if (r < -1.f) return (-420.f + r*(-900.f 
                        + (-540.f - 100.f*r)*r))/120.f;
                    else if (r < 0.f) return (-120.f + r*r*(360.f 
                        + 200.f*r))/120.f;
                    else if (r < 1.f) return  (-120.f + (360.f 
                        - 200.f*r)*r*r)/120.f;
                    else if (r < 2.f) return  (-420.f + r*(900.f 
                        + r*(-540.f + 100.f*r)))/120.f;
                    else if (r < 3.f) return (540.f + r*(-540.f 
                        + (180.f - 20.f*r)*r))/120.f;
                    else return 0.f;
                }
                const char *name(void) const { return "dd-bspline5"; }
            };

            class bspline7: public ::kernel::generating::cloner_base<bspline7> {
            public:
                bool normalize(void) const { return false; }
                bool primal(void) const { return false; }
                int support(void) const { return 8; }
                float value(float r) const {
                    if (r < -4.f) return 0.f;
                    else if (r < -3.f) return (43008 + r*(53760 + r*(26880 
                        + r*(6720 + r*(840 + 42*r)))))/5040.f;
                    else if (r < -2.f) return (-38640 + r*(-82320 
                        + r*(-63840 + r*(-23520 + (-4200 - 294*r)*r))))/5040.f;
                    else if (r < -1.f) return (-1008 + r*(11760 + r*(30240 
                        + r*(23520 + r*(7560 + 882*r)))))/5040.f;
                    else if (r < 0.f) return (-3360 + r*r*(6720 + (-4200 
                        - 1470*r)*r*r))/5040.f;
                    else if (r < 1.f) return (-3360 + r*r*(6720 + r*r*(-4200 
                        + 1470*r)))/5040.f;
                    else if (r < 2.f) return (-1008 + r*(-11760 + r*(30240 
                        + r*(-23520 + (7560 - 882*r)*r))))/5040.f;
                    else if (r < 3.f) return (-38640 + r*(82320 + r*(-63840 
                        + r*(23520 + r*(-4200 + 294*r)))))/5040.f;
                    else if (r < 4.f) return (43008 + r*(-53760 + r*(26880 
                        + r*(-6720 + (840 - 42*r)*r))))/5040.f;
                    else return 0.f;
                }
                const char *name(void) const { return "dd-bspline7"; }
            };
        }

        class box: public cloner_base<box> { // also bspline0 
        public:
            bool normalize(void) const { return false; }
            int support(void) const { return 1; }
            bool primal(void) const { return true; }
            float value(float r) const { 
                if (r < -.5) return 0.f;
                else if (r == -.5) return .5f;
                else if (r < .5) return 1.f;
                else if (r == .5) return .5f;
                else return 0.f;
            }
            const char *name(void) const { return "box"; }
        };

        class hat: public cloner_base<hat> { // also bspline1
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 2; }
            float value(float r) const { 
                r = fabsf(r);
                if (r < 1.f) return 1.f-r;
                else return 0.f;
            }
            const char *name(void) const { return "hat"; }
        };

        // A good reference for the bspline kernels is
        // "B-Spline Signal Processing", Michael Unser, Akram
        // Aldroubi and Murray Eden, 1993, or the earlier
        // "Fast B-Spline Transforms for Continuous Image
        // Representation and Interpolation", Michael Unser,
        // Akram Aldroubi, and Murray Eden, 1991
        class bspline2: public cloner_base<bspline2> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 3; }
            float value(float r) const {
                r = fabsf(r);
                if (r < 0.5f) return (6.f-8.f*r*r)/8.f;
                else if (r < 1.5f) return (9.f+r*(-12.f+4.f*r))/8.f;
                else return 0.f;
            }
            const char *name(void) const { return "bspline2"; }
        };

        class bspline3: public cloner_base<bspline3> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 4; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (4.f + r*r*(-6.f + 3.f*r))/6.f;
                else if (r < 2.f) return  (8.f + r*(-12.f + (6.f - r)*r))/6.f;
                else return 0.f;
            }
            const char *name(void) const { return "bspline3"; }
            base *new_d(void) const { return new d::bspline3; } 
            base *new_dd(void) const { return new dd::bspline3; }
        };

        class bspline4: public cloner_base<bspline4> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 5; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f) return (230.f + r*r*(-240.f + 96.f*r*r))/384.f;
                else if (r < 1.5f) return (220.f + r*(80.f + r*(-480.f
                    + (320.f - 64.f*r)*r)))/384.f;
                else if (r < 2.5f) return (625.f + r*(-1000.f
                    + r*(600.f + r*(-160.f + 16.f*r))))/384.f;
                else return 0.f;
            }
            const char *name(void) const { return "bspline4"; }
        };

        class bspline5: public cloner_base<bspline5> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (66.f + r*r*(-60.f
                            + (30.f - 10.f*r)*r*r))/120.f;
                else if (r < 2.f) return (51.f + r*(75.f + r*(-210.f
                                + r*(150.f + r*(-45.f + 5.f*r)))))/120.f;
                else if (r < 3.f) return (243.f + r*(-405.f + r*(270.f
                                + r*(-90.f + (15.f - r)*r))))/120.f;
                else return 0.f;
            }
            const char *name(void) const { return "bspline5"; }
            base *new_d(void) const { return new d::bspline5; } 
            base *new_dd(void) const { return new dd::bspline5; }
        };

        class bspline7: public cloner_base<bspline7> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 8; }
            float value(float r) const {
                if (r < -4.f) return 0.f;
                else if (r < -3.f) return (16384 + r*(28672 + r*(21504 
                    + r*(8960 + r*(2240 + r*(336 + r*(28 + r)))))))/5040.f;
                else if (r < -2.f) return  (-1112 + r*(-12152 + r*(-19320 
                    + r*(-13720 + r*(-5320 + r*(-1176 
                    + (-140 - 7*r)*r))))))/5040.f;
                else if (r < -1.f) return (2472 + r*(392 + r*(-504 + r*(1960 
                    + r*(2520 + r*(1176 + r*(252 + 21*r)))))))/5040.f;
                else if (r < 0.f) return (2416 + r*r*(-1680 + r*r* (560 
                    + (-140 - 35*r)*r*r)))/5040.f;
                else if (r < 1.f) return (2416 + r*r*(-1680 + r*r* (560 
                    + r*r*(-140 + 35*r))))/5040.f;
                else if (r < 2.f) return (2472 + r*(-392 + r*(-504 + r* (-1960 
                    + r*(2520 + r*(-1176 + (252 - 21*r)*r))))))/5040.f;
                else if (r < 3.f) return (-1112 + r*(12152 + r*(-19320 
                    + r*(13720 + r*(-5320 + r*(1176 + r*(-140 
                    + 7*r)))))))/5040.f;
                else if (r < 4.f) return (16384 + r*(-28672 + r*(21504 
                    + r*(-8960 + r*(2240 + r*(-336 + (28 - r)*r))))))/5040.f;
                else return 0.f;
            }
            const char *name(void) const { return "bspline7"; }
            base *new_d(void) const { return new d::bspline7; } 
            base *new_dd(void) const { return new dd::bspline7; }
        };



        // "Theory and design of local interpolators", A. Schaum, 1993
        // (these are localized Lagrange interpolators)
        class schaum2: public cloner_base<schaum2> { // same as I-MOMS-2
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 3; }
            float value(float r) const {
                r = fabsf(r);
                if (r < 0.5f) return (16.f*(1.f - r*r))/16.f;
                else if (r < 1.5f) return (16.f + r*(-24.f + 8.f*r))/16.f; 
                else return 0.f;
            }
            const char *name(void) const { return "schaum2"; }
        };

        // "Theory and design of local interpolators", A. Schaum, 1993
        // (these are localized Lagrange interpolators)
        class schaum3: public cloner_base<schaum3> { // same as I-MOMS-3
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 4; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (6.f + r*(-3.f + r*(-6.f + 3.f*r)))/6.f;
                else if (r < 2.f) return (6.f + r*(-11.f + (6.f - r)*r))/6.f; 
                else return 0.f;
            }
            const char *name(void) const { return "schaum3"; }
        };

        // "Theory and design of local interpolators", A. Schaum, 1993
        // (these are localized Lagrange interpolators)
        class schaum4: public cloner_base<schaum4> { // same as I-MOMS-4
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 5; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f) return (768.f + r*r*(-960.f + 192*r*r))/768.f;
                else if (r < 1.5f) return (768.f + r*(-640.f + r*(-640.f 
                    + (640.f - 128.f*r)*r)))/768.f;
                else if (r < 2.5f) return (768.f + r*(-1600.f + r*(1120.f 
                    + r*(-320.f + 32.f*r))))/768.f;
                else return 0.f;
            }
            const char *name(void) const { return "schaum4"; }
        };

        // "Theory and design of local interpolators", A. Schaum, 1993
        // (these are localized Lagrange interpolators)
        class schaum5: public cloner_base<schaum5> { // same as I-MOMS-5
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (120.f + r*(-40.f + r*(-150.f 
                    + r*(50.f + (30.f - 10.f*r)*r))))/120.f;
                else if (r < 2.f) return (120.f + r*(-130.f + r*(-75.f 
                    + r*(125.f + r*(-45.f + 5.f*r)))))/120.f;
                else if (r < 3.f) return (120.f + r*(-274.f + r*(225.f 
                    + r*(-85.f + (15.f - r)*r))))/120.f;
                else return 0.f;
            }
            const char *name(void) const { return "schaum5"; }
        }; 

        // "Reconstruction filters in Computer Graphics", Don P. Mitchell 
        // and Arun N. Netravali, 1988
        class mitchell_netravali: public cloner_base<mitchell_netravali> { 
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 4; }
            float value(float r) const {
                r = fabsf(r);
                if (r < 1.f) return (16.f+r*r*(-36.f+21*r))/18.f;
                else if (r < 2.f) return  (32.f+r*(-60.f+(36.f-7.f*r)*r))/18.f;
                else return 0.f;
            }
            const char *name(void) const { return "mitchell-netravali"; }
        };

        // "Cubic Convolution Interpolation for Digital Image Processing", 
        // Robert G. Keys, 1981
        class keys4: public cloner_base<keys4> { // same as Catmull-Rom
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 4; }
            float value(float r) const {
                r = fabsf(r);
                if (r < 1.0f) return (2.f + r*r*(-5.f + 3.f*r))/2.f;
                else if (r < 2.0f) return (4.f + r*(-8.f + (5.f - r)*r))/2.0f;
                else return 0.f;
            }
            const char *name(void) const { return "keys4"; }
        };

        // "Cubic Convolution Interpolation for Digital Image Processing", 
        // Robert G. Keys, 1981
        class keys6: public cloner_base<keys6> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabsf(r);
                if (r < 1.f) return (12.f + r*r*(-28.f + 16.f*r))/12.f;
                else if (r < 2.f) return (30.f + r*(-59.f + (36.f 
                        - 7.f*r)*r))/12.f;
                else if (r < 3.f) return (-18.f + r*(21.f + (-8.f + r)*r))/12.f;
                else return 0.f;
            }
            const char *name(void) const { return "keys6"; }
        };

        // parabola that is zero at 1 and -1, and integrates to 1
        class parabolic: public cloner_base<parabolic> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 2; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.0f) return .75f*(1.f-r*r);
                else return 0.f;
            }
            const char *name(void) const { return "parabolic"; }
        };

        // "Quadratic Interpolation for Image Resampling", Neil A. Dodgson, 1997
        class dogson_interpolation: public cloner_base<dogson_interpolation> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 3; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f)  return -2.f*r*r + 1.f;
                else if (r < 1.5f) return r*r - 2.5f*r + 1.5f; 
                else return 0.f; 
            }
            const char *name(void) const { return "dogson-interpolation"; }
        };
        
        // "Quadratic Interpolation for Image Resampling", Neil A. Dodgson, 1997
        class dogson_approximation: public cloner_base<dogson_approximation> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 3; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f)  return -r*r + .75f; 
                else if (r < 1.5f)  return .5f*r*r - 1.5f*r + 9.f/8.f;
                else return 0.f; 
            }
            const char *name(void) const { return "dogson-approximation"; }
        };

        // "Image Reconstruction by Convolution with Symmetrical
        // Piecewise nth-Order Polynomial Kernels", Erik H. W.
        // Meijering, Karel J. Zuiderveld, Max A. Viergever, 1999 
        class meijering5: public cloner_base<meijering5> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (64.f+r*r*(-136.f+(126.f-54.f*r)*r*r))/64.f;
                else if (r < 2.f) return (122.f+r*(-165.f+r*(-56.f
                    +r*(170.f+r*(-84.f+13.f*r)))))/64.f;
                else if (r < 3.f) return (-486.f+r*(891.f+r*(-648.f
                    +r*(234.f+r*(-42.f+3.f*r)))))/64.f;
                else return 0.f;
            }
            const char *name(void) const { return "meijering5"; }
        };
        
        // "Image Reconstruction by Convolution with Symmetrical
        // Piecewise nth-Order Polynomial Kernels", Erik H. W.
        // Meijering, Karel J. Zuiderveld, Max A. Viergever, 1999 
        class meijering7: public cloner_base<meijering7> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 8; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (83232.f+r*r*(-173328.f+r*r*(134200.f
                    +r*r*(-66117.f+22013.f*r))))/83232.f;
                else if (r < 2.f) return (6216.f+r*(438956.f+r*(-1189728.f
                    +r*(1193220.f+r*(-558240.f+r*(114996.f+(-4293.f
                    -1127.f*r)*r))))))/83232.f;
                else if (r < 3.f) return (2744520.f+r*(-7935956.f+r*(9558912.f
                    +r*(-6252540.f+r*(2408920.f+r*(-548436.f+(68493.f
                    -3627.f*r)*r))))))/83232.f;
                else if (r < 4.f) return (872448.f+r*(-1599488.f+r*(1254144.f
                    +r*(-545280.f+r*(142000.f+r*(-22152.f+(1917.f
                    -71.f*r)*r))))))/83232.f;
                else return 0.f;
            }
            const char *name(void) const { return "meijering7"; }
        };

        // "Short Kernel Fifth-Order Interpolaton", Ismail German, 1997
        class german: public cloner_base<german> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 8; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (144.f+r*r*(-335.f+r*(185.f+6.f*r)))/144.f;
                else if (r < 2.f) return (312.f+r*(-580.f+r*(306.f+(-29.f
                    -9.f*r)*r)))/144.f;
                else if (r < 3.f) return (-48.f+r*(4.f+r*(40.f+r*(-21.f
                    +3.f*r))))/144.f;
                else if (r < 4.f) return (-48.f+r*(40.f+(-11.f+r)*r))/144.f;
                else return 0.f;
            }
            const char *name(void) const { return "meijering5"; }
        };

        // "Linear Interpolation Revitalized", Thierry Blu,
        // Philippe Thévenaz, Michael Unser, 2004
        class linrev: public cloner_base<linrev> { 
        public:
            bool normalize(void) const { return false; }
            int support(void) const { return 3; }
            bool primal(void) const { return true; }
            float value(float r) const { 
                static const float tau = 0.5f*(1.f - sqrtf(3.f)/3.f);
                r = fabs(r - tau);
                if (r < 1.f) return 1.f-r;
                else return 0.f;
            }
            const char *name(void) const { return "linrev"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class omoms2: public cloner_base<omoms2> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 3; }
            float value(float r) const {
                r = fabsf(r);
                if (r < .5f) return (86.f-120.f*r*r)/120.f;
                else if (r == .5f) return 58.f/120.f; // discontinuity!
                else if (r < 1.5f) return (137.f+r*(-180.f+60*r))/120.f;
                else return 0.f;
            }
            const char *name(void) const { return "omoms2"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class omoms3: public cloner_base<omoms3> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 4; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (26.f+r*(3.f+r*(-42.f+21.f*r)))/42.f;
                else if (r < 2.f) return  (58.f+r*(-85.f+(42.f-7.f*r)*r))/42.f;
                else return 0.f;
            }
            const char *name(void) const { return "omoms3"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class omoms4: public cloner_base<omoms4> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 5; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f) return (68298.f+r*r*(-65520.f
                    +30240.f*r*r))/120960.f;
                else if (r == 0.5f) return 53776.f/120960.f; // discontinuity!
                else if (r < 1.5f) return (60868.f+r*(42000.f+r*(-157920.f
                    +(100800.f-20160.f*r)*r)))/120960.f;
                else if (r == 1.5f) return 6696/120960.f; // discontinuity!
                else if (r < 2.5f) return (207383.f+r*(-323400.f+r*(190680.f
                    +r*(-50400.f+5040.f*r))))/120960.f;
                else return 0.f;
            }
            const char *name(void) const { return "omoms4"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class omoms5: public cloner_base<omoms5> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (4122.f+r*(-10.f+r*(-3240.f+r*(-400.f
                    +(1980.f-660.f*r)*r))))/7920.f;
                else if (r < 2.f) return (2517.f+r*(6755.f+r*(-14940.f
                    +r*(10100.f +r*(-2970.f+330.f*r)))))/7920.f;
                else if (r < 3.f) return (17121.f+r*(-27811.f+r*(18180.f
                    +r*(-5980.f+(990.f-66.f*r)*r))))/7920.f;
                else return 0.f;
            }
            const char *name(void) const { return "omoms5"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class somoms4: public cloner_base<somoms4> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return true; }
            int support(void) const { return 5; }
            float value(float r) const {
                r = fabs(r);
                if (r < 0.5f) return (1090.f+r*r*(-1056.f+480.f*r*r))/1920.f;
                else if (r < 1.5f) return (980.f+r*(640.f+r*(-2496.f+(1600.f
                    -320.f*r)*r)))/1920.f;
                else if (r < 2.5f) return (3275.f+r*(-5120.f+r*(3024.f
                    +r*(-800.f+80.f*r))))/1920.f;
                else return 0.f;
            }
            const char *name(void) const { return "somoms4"; }
        };

        // "MOMS: Maximal-Order Interpolation of Minimal Support", 
        // Thierry Blu, Philippe Thévenaz, Michael Unser, 2001
        class somoms5: public cloner_base<somoms5> {
        public:
            bool normalize(void) const { return false; }
            bool primal(void) const { return false; }
            int support(void) const { return 6; }
            float value(float r) const {
                r = fabs(r);
                if (r < 1.f) return (6474.f+r*r*(-5760.f+r*(-100.f
                    +(2970.f-990.f*r)*r)))/11880.f;
                else if (r < 2.f) return (4839.f+r*(7875.f+r*(-21060.f
                    +r*(14900.f+r*(-4455.f+495.f*r)))))/11880.f;
                else if (r < 3.f) return (24327.f+r*(-40365.f+r*(26820.f
                    +r*(-8920.f+(1485.f-99.f*r)*r))))/11880.f;
                else return 0.f;
            }
            const char *name(void) const { return "somoms5"; }
        };

        // CRT Spot from "Filtering high quality text for
        // display on raster scan devices",
        // J Kajiya, M Ullner, 1981
        class crt: public cloner_base<crt> {
        public:
            bool normalize(void) const { return true; }
            bool primal(void) const { return true; }
            int support(void) const { return 6; }
            float value(float r) const {
                return -0.000023979224f + 0.59850358f*expf(-1.125f*r*r);
            }
            const char *name(void) const { return "crt"; }
        };

    } // namespace geneating

} // namespace kernel

#endif
