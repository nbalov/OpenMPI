/*
 * Matrix-vector multiplication, Version 1
 * Michael Quinn
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "MyMPI.h"

typedef double dtype;
#define mpitype MPI_DOUBLE

int main(int argc, char *argv[]) {
	dtype **a;		/* first factor, a matrix */
	dtype *b;		/* second factor, a vector */
	dtype *c_block;		/* partial product, a vector */
	dtype *c;		/* replicated product */
	dtype *storage;		/* matrix elements stored here */
	int i, j;
	int id, p;
	int m, n;
	int nprime;
	int rows;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &id);
	MPI_Comm_size (MPI_COMM_WORLD, &p);

	if (id == 0) {
		if (argc < 5 || argv[1] == NULL || argv[2] == NULL || 
			argv[3] == NULL || argv[4] == NULL) {
			printf ("Enter a valid filename.\n");
			fflush (stdout);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		write_random_matrix (argv[1], mpitype, 
			(int)atoi(argv[3]), (int)atoi(argv[4]));	
		write_random_vector (argv[2], mpitype, (int)atoi(argv[4]));
	}

	read_row_striped_matrix (argv[1], (void*)&a, 
		(void*)&storage, mpitype, &m, &n, MPI_COMM_WORLD);
	rows = BLOCK_SIZE (id, p, m);
	print_row_striped_matrix ((void**)a, mpitype, m, n, 
		MPI_COMM_WORLD);

	read_replicated_vector (argv[2], (void**)&b, mpitype, 
		&nprime, MPI_COMM_WORLD);
	print_replicated_vector (b, mpitype, nprime, 
		MPI_COMM_WORLD);

	c_block = (dtype*) malloc (rows*sizeof(dtype));
	c = (dtype*) malloc (m*sizeof(dtype));

	for (i = 0; i < rows; i++) {
		c_block[i] = 0;
		for (j = 0; j < n; j++) {
			c_block[i] += a[i][j]*b[j];
		}
	}

	replicate_block_vector (c_block, m, (void*)c, mpitype, 
		MPI_COMM_WORLD);

	print_replicated_vector (c, mpitype, m, MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}


