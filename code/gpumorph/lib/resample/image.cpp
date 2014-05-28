#include <cstring>
#include <cctype>
#include "extension.h"

#include "image.h"
#include "error.h"

namespace nehab
{

namespace image {

    int load(const rod::dimage<float>& image, image::rgba<float> *rgba)
    {
        if (image.height()!=0&&image.width()!=0)
        {
            std::vector<float> cpu;
            image.copy_to_host(cpu);

            int width=image.width();
            int height=image.height();
            rgba->resize(height, width);
#pragma omp parallel for
            for (int i = height-1; i >= 0; i--) {
                for (int j = 0; j < width; j++) {
                    int p = i*width+j; // flip image so y is up
                    rgba->r[p] = rgba->g[p] = rgba->b[p] = color::srgbuncurve(cpu[p]/255.f);
                    rgba->a[p] = 1.0f;
                }
            }
            return 1;
        }
        else
        {
            return 0;
        }
    }

    int store(rod::dimage<float>& image, const image::rgba<float> &rgba)
    {
        std::vector<float> cpu(rgba.height()*rgba.width());

        static extension::clamp clamp;

        int height=rgba.height();
        int width=rgba.width();
#pragma omp parallel for
        for (int i = height-1; i >= 0; i--) {
            for (int j = 0; j < width; j++) {
                int p = i*width+j; // flip image so y is up
                assert(rgba.r[p] == rgba.g[p]);
                assert(rgba.g[p] == rgba.b[p]);

                cpu[p] = color::srgbcurve(clamp(rgba.r[p]))*255.f;
            }
        }

        image.copy_from_host(cpu, image.width(), image.height());

        return 1;
    }

}

}
