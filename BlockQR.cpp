#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <omp.h>
#include <mkl.h>
#include <Utils.hpp>

using namespace std;

#define MAX_LC 10

int main(const int argc, const char **argv)
{
    // Usage "a.out [size of matrix: m ]"
    if (argc < 2)
    {
        cerr << "usage: a.out [size of matrix: m ]\n";
        return EXIT_FAILURE;
    }

    const int m = atoi(argv[1]);       // # rows and columns <- square matrix

    double* A = new double [m*m];      // Original matrix
    double* OA = new double[m*m];      // OA: copy of A
    const int lda = m;                 // Leading dimension of A

	double* tau = new double[m];       // elementary refrector vector
    /////////////////////////////////////////////////////////

	double timer;

    Gen_rand_mat((unsigned) time(NULL), m, m, OA);   // Randomize elements of orig. matrix


	for (int lc=0; lc<MAX_LC; lc++)
	{
		cblas_dcopy(m*m, OA, 1, A, 1);
		timer = omp_get_wtime();       // Timer start

		LAPACKE_dgeqrf(MKL_COL_MAJOR, m, m, A, lda, tau);

		timer = omp_get_wtime() - timer; // Timer stop
		cout << m << ", " << timer << endl;
	}

    delete [] A;
    delete [] OA;
	delete [] tau;

	return EXIT_SUCCESS;
}

