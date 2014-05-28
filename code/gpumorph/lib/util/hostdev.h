#ifndef ROD_HOSTDEV_H
#define ROD_HOSTDEV_H

#if !defined(HOSTDEV)
#   if defined(__CUDA_ARCH__)
#       define HOSTDEV __host__ __device__
#   else
#       define HOSTDEV
#   endif
#endif

#endif
