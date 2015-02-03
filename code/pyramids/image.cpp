#include <cstring>
#include <cctype>
#include "extension.h"

#include "image.h"
#include "error.h"
static extension::clamp clamp;

namespace image {

        template<>
 	int load(cv::Mat& image, image::rgba<float> *rgba) {

		if (image.rows!=0&&image.cols!=0)
		{
			int width=image.cols;
			int height=image.rows;
			rgba->resize(height, width);
			const float tof = (1.f/255.f);
			#pragma omp parallel for
			for (int i = height-1; i >= 0; i--) {
				for (int j = 0; j < width; j++) {
					int p = i*width+j; // flip image so y is up
					Vec3b v=image.at<Vec3b>(i,j);
					rgba->r[p] = color::srgbuncurve(v[2]*tof);
					rgba->g[p] = color::srgbuncurve(v[1]*tof);
					rgba->b[p] = color::srgbuncurve(v[0]*tof);
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

        template<>
  	int store(cv::Mat& image, const image::rgba<float> &rgba) {

		int height=rgba.height();
		int width=rgba.width();
		#pragma omp parallel for
		for (int i = height-1; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				int p = i*width+j; // flip image so y is up
				float r = color::srgbcurve(clamp(rgba.r[p]));
				float g = color::srgbcurve(clamp(rgba.g[p]));
				float b = color::srgbcurve(clamp(rgba.b[p]));
				Vec3b v;
				v[0]=(uchar)(255.f*b);
				v[1]=(uchar)(255.f*g);
				v[2]=(uchar)(255.f*r);
				image.at<Vec3b>(i,j)=v;
			}
		}
        return 1;
 }

}
