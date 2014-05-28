#ifndef GPUMORPH_INIT_H
#define GPUMORPH_INIT_H

#include <util/linalg_fwd.h>
#include <vector_types.h>

struct PyramidLevel;

typedef rod::Matrix<int,5,5> imat5;
typedef rod::Matrix<int,3,3> imat3;
typedef rod::Matrix<float,5,5> fmat5;

void calc_tps_stencil(rod::Matrix<fmat5,5,5> &tps);

void calc_nb_improvmask_check_stencil(PyramidLevel &lvl,
                                      rod::Matrix<imat3,5,5> &mask,
                                      rod::Matrix<int,3,3> &offsets);

void calc_nb_io_stencil(PyramidLevel &lvl, rod::Matrix<imat5,5,5> &mask);

#endif
