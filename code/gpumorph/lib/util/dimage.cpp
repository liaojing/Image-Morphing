#include <cuda.h>
#include "dimage.h"
#include "image_ops.h"

namespace rod
{

template <class T>
void copy_to_array(cudaArray *out, dimage_ptr<const T> in)
{
    typedef typename dimage_ptr<const T>::texel_type textype;
    cudaMemcpy2DToArray(out, 0, 0, in,
                        in.rowstride()*sizeof(textype),
                        in.width()*sizeof(textype), in.height(),
                        cudaMemcpyDeviceToDevice);
}

template void copy_to_array(cudaArray *out, dimage_ptr<const float> img);
template void copy_to_array(cudaArray *out, dimage_ptr<const float2> img);
template void copy_to_array(cudaArray *out, dimage_ptr<const float3> img);
template void copy_to_array(cudaArray *out, dimage_ptr<const float4> img);

template <class T>
void copy_from_array(dimage_ptr<T> out, const cudaArray *in)
{
    typedef typename dimage_ptr<const T>::texel_type textype;

    cudaMemcpy2DFromArray(&out, out.rowstride()*sizeof(textype),
                          in, 0, 0, out.width()*sizeof(textype), out.height(),
                          cudaMemcpyDeviceToDevice);
}

template void copy_from_array(dimage_ptr<float> out, const cudaArray *in);
template void copy_from_array(dimage_ptr<float2> out, const cudaArray *in);
template void copy_from_array(dimage_ptr<float3> out, const cudaArray *in);
template void copy_from_array(dimage_ptr<float4> out, const cudaArray *in);

}
