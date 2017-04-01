/*
 * Floyd's all-pairs shortest-path algorithm
 * see Ch.6 in M. J. Quinn
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

void compute_shortest_paths (int, int, dtype**, int);

int main (int argc, char *argv[]) {

	dtype **a;	/* doubly-subscripted array */
	dtype  *storage;/* local portion of array elements */
	int i, j, k;
	int id;		/* process rank */
	int p;		/* number of processes */
	int m, n;	/* matrix dimensions */

	//write_random_adjacency_matrix(argv[1], MPI_TYPE, 
	//	(int)atoi(argv[2]), (int)atoi(argv[3]));
	//return 0;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &id);
	MPI_Comm_size (MPI_COMM_WORLD, &p);

	if (id == 0) {
		if (argv[1] == NULL) {
			printf("Enter a valid filename.\n");
			fflush(stdout);
		}
		if (argv[2] != NULL && argv[3] != NULL) {
			write_random_adjacency_matrix(argv[1], MPI_TYPE, 
				(int)atoi(argv[2]), (int)atoi(argv[3]));
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	read_row_striped_matrix (argv[1], (void*)&a, (void*)&storage, 
		MPI_TYPE, &m, &n, MPI_COMM_WORLD);

	if (m != n) terminate(id, "Matrix must be square\n");

	print_row_striped_matrix ((void**)a, MPI_TYPE, m, n, 
		MPI_COMM_WORLD);

	compute_shortest_paths (id, p, (dtype**)a, n);
	
	print_row_striped_matrix ((void**)a, MPI_TYPE, m, n, 
		MPI_COMM_WORLD);

	MPI_Finalize();

	return 0;
}

void compute_shortest_paths (int id, int p, dtype **a, int n) {

	int i, j, k;
	int offset;	/* local index to broadcast row */
	int root;	/* process controlling row to be bcast */
	int *tmp;	/* holds the broadcast row */

	tmp = (dtype*)malloc(n*sizeof(dtype));
	for (k = 0; k < n; k++) {
		root = BLOCK_OWNER(k,p,n);
		if (root == id) {
			offset = k - BLOCK_LOW(id,p,n);
			for (j = 0; j < n; j++) {
				tmp[j] = a[offset][j];
			}
		}
		MPI_Bcast(tmp, n, MPI_TYPE, root, MPI_COMM_WORLD);
		for (i = 0; i < BLOCK_SIZE(id,p,n); i++) {
			for (j = 0; j < n; j++) {
				if ((int)a[i][j] < RAND_MAX & 
				    (int)a[i][k] < RAND_MAX & 
				    (int)tmp[j] < RAND_MAX) {
					a[i][j] = MIN(a[i][j], a[i][k]+tmp[j]);
				}
			}
		}
	}
	free (tmp);
}

