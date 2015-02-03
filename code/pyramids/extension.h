#ifndef EXTENSION_H
#define EXTENSION_H

#include <cmath>

namespace extension {

    class base {
    public:
        virtual ~base() { ; }
		virtual float operator()(float t) const = 0; 
		virtual int operator()(int i, int n) const = 0; 
        virtual float wrap(float t) const { return (*this)(t); }
        virtual int wrap(int i, int n) const { return (*this)(i, n); }
    };

    class none: public base {
    public:
		float operator()(float t) const {
			return t;
		}
		int operator()(int i, int n) const {
            (void) n;
			return i;
		}
    };

    class clamp: public base {
    public:
		float operator()(float t) const {
			if (t < 0.f) return 0.f;
			else if (t > 1.f) return 1.f;
			else return t;
		}
		int operator()(int i, int n) const {
			if (i < 0) return 0;
			else if (i >= n) return n-1;
			else return i;
		}
    };

    class repeat: public base {
    public:
		float operator()(float t) const {
			t = fmodf(t, 1.f);
			return t < 0.f? t+1.f: t;
		}
		int operator()(int i, int n) const {
			if (i >= 0) return i % n;
			else return (n-1) - ((-i-1) % n);
        }
	};

    class mirror: public base {
    public:
		float operator()(float t) const {
			t = fabs(fmodf(t, 2.f));
			return t > 1.f? 2.f-t: t;
		}
		int operator()(int i, int n) const {
            repeat r;
			i = r(i, 2*n); 
            if (i >= n) return i = (2*n)-i-1;
            else return i;
        }
	};
}

#endif
