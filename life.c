/*
 * Game of Life, John Conway, 1970
 * see Ex. 6.13 in M. J. Quinn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

void iterate (int, int, dtype**, int, int, int, int);

void write3x3blinker(char *filename) {
	int i;
	dtype *buff;
	buff = (dtype*)malloc (9*sizeof(dtype));
	if (!buff) return;
	memset(buff, 0, 9*sizeof(dtype));
	buff[3] = buff[4] = buff[5] = 1;
	write_buffered_matrix(filename, MPI_TYPE, buff, 3, 3);
}

void write9x9glider(char *filename) {
	int i;
	dtype *buff;
	buff = (dtype*)malloc (81*sizeof(dtype));
	if (!buff) return;
	memset(buff, 0, 81*sizeof(dtype));
	buff[0] = 1;
	buff[10] = buff[11] = 1;
	buff[18] = buff[19] = 1;
	write_buffered_matrix(filename, MPI_TYPE, buff, 9, 9);
}

int main (int argc, char *argv[]) {

	dtype **a;	/* doubly-subscripted array */
	dtype  *storage;/* local portion of array elements */
	int k;
	int id;		/* process rank */
	int p;		/* number of processes */
	int m, n;	/* matrix dimensions */
	int niter = 0;  /* number of iterations */
	int nevery= 0;  /* print state every # iterations */
	char filename[256], *pfile;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &id);
	MPI_Comm_size (MPI_COMM_WORLD, &p);

	pfile = argv[1];
	if (id == 0 && argc > 5) {
		if (argc < 6) {
			terminate (id, "Usage: <filename> <m> <n> <numiter> <print_every>");
		}
		if (argv[1] == NULL) {
			printf("Enter a valid filename.\n");
			fflush(stdout);
		}
		if (argv[2] != NULL && argv[3] != NULL) {
			write_random_binary_matrix(argv[1], MPI_TYPE, 
				(int)atoi(argv[2]), (int)atoi(argv[3]));
		}
		if (argv[4] != NULL) {
			niter = (int)atoi(argv[4]);
		}
		if (argv[5] != NULL) {
			nevery = (int)atoi(argv[5]);
		}
		if (niter < 1 || nevery < 1 || nevery > niter) {
			terminate (id, "Invalid <numiter> and <print_every> arguments.");
		}
	}
	else if (id == 0) {
		//sprintf(filename, "glider9x9.mat");
		//printf("%d: write file %s\n", filename);
		//pfile = &filename[0];
		write9x9glider(pfile);
		if (argc < 4) {
			terminate (id, "Usage: <filename> <m> <n> <numiter> <print_every>");
		}
		if (argv[2] != NULL) {
			niter = (int)atoi(argv[2]);
		}
		if (argv[3] != NULL) {
			nevery = (int)atoi(argv[3]);
		}
		if (niter < 1 || nevery < 1 || nevery > niter) {
			terminate (id, "Invalid <numiter> and <print_every> arguments.");
		}
		
	}

	MPI_Bcast(&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nevery, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	
	read_row_striped_matrix (pfile, (void*)&a, (void*)&storage, 
		MPI_TYPE, &m, &n, MPI_COMM_WORLD);
	print_row_striped_binary_matrix ((void**)a, MPI_TYPE, m, n, 
		MPI_COMM_WORLD);
	iterate (id, p, (dtype**)a, m, n, niter, nevery);	
	//print_row_striped_binary_matrix ((void**)a, MPI_TYPE, m, n, 
	//	MPI_COMM_WORLD);

	MPI_Finalize();

	return 0;
}

void iterate (int id, int p, dtype **a, int m, int n, int niter, int nevery) {

	int i, j, k;
	int offset;	/* row offset in a */
	int local_rows;
        dtype *tmp;
	dtype **btmp;	/* holds the rows of current state */
	void  *ptr;
	MPI_Status status;
	int cnt;	/* number of live neighbours */

	local_rows = BLOCK_SIZE (id,p,m);
	offset     = BLOCK_LOW  (id,p,m);
	
	tmp  = (dtype*) malloc ((local_rows+2)*n*sizeof(dtype));
        btmp = (dtype**)malloc ((local_rows+2)*PTR_SIZE);
	ptr = tmp;
	for (k = 0; k < local_rows+2; k++) {
		btmp[k] = ptr;
		ptr += n*sizeof(dtype);
	}

	for (k = 0; k < niter; k++) {

	// rows btmp[1,local_rows] contain a[0,local_rows]
	memcpy((void*)btmp[1], (void*)a[0], local_rows*n*sizeof(dtype));

	if (id < p-1) {
		MPI_Send(btmp[local_rows], n, MPI_TYPE, id+1, DATA_MSG, 
			MPI_COMM_WORLD);
	}
        if (id > 0) {
                MPI_Send(btmp[1], n, MPI_TYPE, id-1, DATA_MSG,
                        MPI_COMM_WORLD);
        }

	if (id == 0) {
		memset ((void*)btmp[0], 0, n*sizeof(dtype));
	}
	else {
		MPI_Recv(btmp[0], n, MPI_TYPE, id-1, DATA_MSG, 
			MPI_COMM_WORLD, &status);
	}

	if (id == p-1) {
		memset ((void*)btmp[local_rows+1], 0, n*sizeof(dtype));
	}
	else {
		MPI_Recv(btmp[local_rows+1], n, MPI_TYPE, id+1, DATA_MSG, 
			MPI_COMM_WORLD, &status);
	}
	
	for (i = 0; i < local_rows; i++) {
		for (j = 0; j < n; j++) {
			cnt = btmp[i][j] + btmp[i+2][j];
			if (j > 0) {
				cnt += btmp[i]  [j-1];
				cnt += btmp[i+1][j-1];
				cnt += btmp[i+2][j-1];
			}
			if (j < n-1) {
				cnt += btmp[i]  [j+1];
				cnt += btmp[i+1][j+1];
				cnt += btmp[i+2][j+1];
			}
			if (a[i][j] == 0 && cnt == 3) {
				a[i][j] = 1;
			}
			if (a[i][j] == 1 && (cnt < 2 || cnt > 3)) {
				a[i][j] = 0;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	if ((int)(k % nevery) == 0) {		
		print_row_striped_binary_matrix (
			(void**)a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	}

	} // k

	free (tmp);
	free (btmp);
}

