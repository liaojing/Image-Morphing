#include <sstream>
#include <iostream>
#include <util/timer.h>
#include <util/image_ops.h>
#include <cstdio>
#include "imgio.h"
#include "config.h"
#include "pyramid.h"
#include "morph.h"
#include "param_io.h"
#include "parameters.h"
#if CUDA_SM < 20
#   include <util/cuPrintf.cuh>
#endif

struct callback_data
{
    callback_data(const rod::dimage<float3> &_dimg0,
                  const rod::dimage<float3> &_dimg1)
        : dimg0(_dimg0)
        , dimg1(_dimg1)
    {
    }

    const rod::dimage<float3> &dimg0, &dimg1;
};

bool callback(const std::string &text, int cur, int max,
              const rod::dimage<float2> *halfway,
              const rod::dimage<float> *ssim_error,
              void *data)
{
    static int idx=0;
    if(halfway)
    {
        callback_data *cbdata = reinterpret_cast<callback_data *>(data);


        rod::dimage<float3> result;
        render_halfway_image(result, *halfway, cbdata->dimg0, cbdata->dimg1);

        char buffer[100];
        sprintf(buffer,"hw_%03d.png",idx++);

        std::cout << cur << "/" << max << ": " << buffer << std::endl;

        save(buffer,result);
    }

      //  std::cout << "has halfway" << std::endl;

    //std::cout << cur << "/" << max << ": " << text << std::endl;
    return true;
}

int main(int argc, char *argv[])
{
    try
    {
#if CUDA_SM < 30
        cudaPrintfInit();
#endif

     /*   if(argc <= 1 || argc >= 4)
            throw std::runtime_error("Bad arguments");*/

        // default values defined in default ctor
        Parameters params;

            params.fname0 = "image1.png";
            params.fname1 = "image2.png";
       

        rod::base_timer *morph_timer = NULL;

        if(params.verbose)
            morph_timer = &rod::timers.gpu_add("Total");

        Morph morph(params);

#if 0
        callback_data cbdata(morph.image0(), morph.image1());
        morph.set_callback(&callback, &cbdata);
#endif

        rod::dimage<float2> halfway;

        if(!morph.calculate_halfway_parametrization(halfway))
        {
            if(morph_timer)
                morph_timer->stop();

            throw std::runtime_error("Aborted");
        }

        if(morph_timer)
            morph_timer->stop();

        /* to convert from halfway (a dimage<float2>) to std::vector<float2>,
           do:

           std::vector<float2> host; // will have size halfway.width()*halfway.height()
           halfway.copy_to_host(host);
        */


        rod::dimage<float3> result;

        render_halfway_image(result, halfway, morph.image0(), morph.image1());

        if(params.verbose)
            rod::timers.flush();

        save("result.png",result);

#if CUDA_SM < 30
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
#endif
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
