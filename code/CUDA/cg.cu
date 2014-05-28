#include "cg.cuh"
#include <cusp/krylov/cg.h>

int solve(cusp::csr_matrix<int,float,cusp::host_memory> &A,cusp::array1d<float, cusp::host_memory> &B,cusp::array1d<float, cusp::host_memory> &X)
{
	// create an empty sparse matrix structure (HYB format)
	cusp::csr_matrix<int, float, cusp::device_memory> A_d=A;

	// allocate storage for solution (x) and right hand side (b)
	cusp::array1d<float, cusp::device_memory> X_d=X;
	cusp::array1d<float, cusp::device_memory> B_d=B;

	// set stopping criteria:
	cusp::verbose_monitor<float> monitor(B_d, 5000, 0.001f);

	// set preconditioner (identity)
	cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);

	// solve the linear system A * x = b with the Conjugate Gradient method
	cusp::krylov::cg(A_d, X_d, B_d, monitor, M);

	X=X_d;

	return 0;
}