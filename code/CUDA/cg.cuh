#include <cusp/hyb_matrix.h>
#include <cusp/csr_matrix.h>


int solve(cusp::csr_matrix<int,float,cusp::host_memory> &A,cusp::array1d<float, cusp::host_memory> &B,cusp::array1d<float, cusp::host_memory> &X);
