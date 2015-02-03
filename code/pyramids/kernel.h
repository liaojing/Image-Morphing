#ifndef KERNEL_H 
#define KERNEL_H

#include <cstdio>

#include "generating.h"
#include "discrete.h"
#include "dlti.h"

namespace kernel {

    // base class includes methods from generating functions
    // (used to sample the kernel) and functions to query
    // the discrete filtering components
    class base: public generating::base {
    public:
        virtual ~base() { ; }
        virtual const discrete::base *fir(void) const { return &delta; }
        virtual const discrete::base *ifir(void) const { return &delta; }
        virtual base *clone(void) const = 0; 
    private:
        discrete::delta delta;
    };

    template <class T>
    class cloner_base: public base{
    public:
        virtual base *clone(void) const {
            return new T(static_cast<const T &>(*this));
        }
    };

    static inline void build_name(const discrete::base *fir, 
        const discrete::base *ifir, const generating::base *ker, char *name) {
        if (discrete::isdelta(fir)) {
            if (discrete::isdelta(ifir)) strcpy(name, ker->name()); 
            else sprintf(name, "%.30s*%.30s^(-1)", ker->name(), ifir->name());
        } else if (discrete::isdelta(ifir)) {
            sprintf(name, "%.30s*%.30s", ker->name(), fir->name());
        } else {
            sprintf(name, "%.30s*%.30s*%.30s^(-1)", ker->name(),
                fir->name(), ifir->name());
        }
    }

    // this shouldn't be used for reconstruction! 
    // this kernel stores an approximation of the mixed convolution 
    // between a generating function and given discrete filters.  
    // it simply shows the approximate shape of a generalized kernel
    // it may have a gigantic support and is used only for plots
    class cardinal: public cloner_base<cardinal> {
    public:
        cardinal(base *ker) {
            const float EPS = 0.001f; // tolerance
            const int N = sizeof(m_ir)/sizeof(m_ir[0]); 
            m_ker = ker;
            // create unit impulse of size N
            for (int i = 0; i < N; i++)
                m_ir[i] = 0.f;
            m_ir[N/2] = 1.f;
            // compute impulse response 
            dlti::filter(m_ker->fir(), m_ker->ifir(), m_ir, N);
            // find non-zero range up to tolerance to the right and left
            int l, r; 
            for (l = 0; l < N/2; l++)
                if (fabs(m_ir[l]) > EPS) break;
            for (r = N-1; r > N/2; r--)
                if (fabs(m_ir[r]) > EPS) break;
            r = abs(N/2 - r);
            l = abs(l - N/2);
            m_r = r > l? r: l;
        }
        ~cardinal() { delete m_ker; }
        bool normalize(void) const { return m_ker->normalize(); }
        float value(float t) const {
            const int N = sizeof(m_ir)/sizeof(m_ir[0]); 
            // evaluate the result of mixed convolution between
            // the kernel and the impulse response at position t
            float s = 0.f;
            for (int i = -m_r; i <= m_r; i++) {
                s += m_ir[N/2+i]*m_ker->value(t-i);
            }
            return s;
        }
        bool primal(void) const { return m_ker->primal(); }
        int support(void) const { return m_ker->support() + 2*m_r; }
        const discrete::base *fir(void) const { return &delta; }
        const discrete::base *ifir(void) const { return &delta; }
        const char *name(void) const { return m_ker->name(); }
        generating::base *new_d(void) const { return m_ker->new_d(); }
        generating::base *new_dd(void) const { return m_ker->new_dd(); }
        cardinal(const cardinal& other): cloner_base<cardinal>() {
            m_ker = other.m_ker->clone();
            memcpy(m_ir, other.m_ir, sizeof(m_ir));
            m_r = other.m_r;
        }
    private:
        float m_ir[100];
        int m_r;
        base *m_ker;
        discrete::delta delta;
    };

    // even simpler shortcut
    class simple: public cloner_base<simple> {
    public:
        simple(generating::base *ker) { m_ker = ker; }
        virtual ~simple() { delete m_ker; }
        bool normalize(void) const { return m_ker->normalize(); }
        bool primal(void) const { return m_ker->primal(); }
        int support(void) const { return m_ker->support(); }
        float value(float t) const { return m_ker->value(t); }
        const char *name(void) const { return m_ker->name(); }
        generating::base *new_d(void) const { return m_ker->new_d(); }
        generating::base *new_dd(void) const { return m_ker->new_dd(); }
        simple(const simple& other): cloner_base<simple>() 
            { m_ker = other.m_ker->clone(); }
    private:
        generating::base *m_ker;
    };

    // this is what *should* be used for reconstruction! 
    // it stores the generating function and discrete filters separately  
    // that way, whoever uses the kernel can apply the
    // discrete filter before reconstruction, or after filtering
    class generalized: public cloner_base<generalized> {
    public:
        generalized(discrete::base *fir, discrete::base *ifir, 
            generating::base *ker) { 
            m_name[0] = '\0'; 
            m_fir = fir; 
            m_ifir = ifir;
            m_ker = ker;
        }
        virtual ~generalized() {
            delete m_fir;
            delete m_ifir;
            delete m_ker;
        }
        bool normalize(void) const { return m_ker->normalize(); }
        bool primal(void) const { return m_ker->primal(); }
        int support(void) const { return m_ker->support(); }
        float value(float t) const { return m_ker->value(t); }
        const discrete::base *fir(void) const { return m_fir; }
        const discrete::base *ifir(void) const { return m_ifir; }
        const char *name(void) const { 
            // seldom used function, faster to generate name when needed
            // than always generate in constructor (lazy evaluation)
            if (!m_name[0])
                build_name(m_fir, m_ifir, m_ker, const_cast<char *>(m_name));
            return m_name;
        }
        generating::base *new_d(void) const { return m_ker->new_d(); }
        generating::base *new_dd(void) const { return m_ker->new_dd(); }
        // deep copy constructor
        generalized(const generalized &other): cloner_base<generalized>() {
            m_ker = other.m_ker->clone();
            m_fir = other.m_fir->clone();
            m_ifir = other.m_ifir->clone();
            strcpy(m_name, other.m_name);
        }
    private:
        char m_name[256];
        generating::base *m_ker;
        discrete::base *m_fir;
        discrete::base *m_ifir;
    };

    // to be used with weighted patterns that do not require a prefilter
    class delta: public cloner_base<delta> {
    public:
        bool normalize(void) const { return false; }
        bool primal(void) const { return 0; }
        int support(void) const { return 0; }
        float value(float) const { return 1.f; }
        const char *name(void) const { return "delta"; }
    };

} // namespace kernel

#endif
