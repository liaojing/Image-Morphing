#include "extension.h"
#include "kernel.h"
#include "discrete.h"
#include "error.h"
#include "scale.h"
#include "time.h"

int main(int argc, char *argv[]) {
      if (argc < 2)
         errorf(("pyramid <input>"));
    const char *name = argv[1];
    image::rgba<float> image;
    if (!image::load(name, &image))
        errorf(("unable to load input image '%s'", name));
    int level = 1;
    // use cardinal bspline3 prefilter for downsampling
    kernel::base *pre = new kernel::generalized(
            new kernel::discrete::delta,
            new kernel::discrete::sampled(new kernel::generating::bspline3),
            new kernel::generating::bspline3);
    // no additional discrete processing
    kernel::discrete::base *delta = new kernel::discrete::delta;
    // use mirror extension
    extension::base *ext = new extension::mirror;
    // loop downsampling
	clock_t start, finish;
    start=clock();
    while (image.width() > 8 && image.height() > 8) {
        char new_name[FILENAME_MAX];
        sprintf(new_name, "%d.bmp", level);
        if (!scale((image.height()+1)/2, (image.width()+1)/2, pre, 
            delta, delta, ext, &image, new_name))
            errorf(("error processing level %d", level));
        level++;
    }
	finish=clock();
	float run_time= (float)(finish - start) / CLOCKS_PER_SEC; 
	printf("%f sec",run_time);
    return 0;
}
