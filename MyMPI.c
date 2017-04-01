/*
 * A library of vector/matrix/input/output functions
 * 
 * Programmed by Michael J. Quinn
 * 4 Sep 2002
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "MyMPI.h"

int get_size(MPI_Datatype t) {
	if (t == MPI_BYTE) return sizeof(char);
	if (t == MPI_DOUBLE) return sizeof(double);
	if (t == MPI_FLOAT) return sizeof(float);
	if (t == MPI_INT) return sizeof(int);
	printf("Error: unrecognzed argument to 'get_size'\n");
	fflush(stdout);
	MPI_Abort(MPI_COMM_WORLD, TYPE_ERROR);
}

/* 
 * allocate space from the heap
 */

void *my_malloc(
	int id,		/* IN - process rank */
	int bytes)	/* IN - bytes to allocate */
{
	void *buffer;
	if ((buffer = malloc((size_t)bytes)) == NULL) {
		printf("Error: malloc failed for process %d\n", id);
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
	}
	return buffer;
}

/*
 * all processes must invoke this function together, but 
 * process 0 prints the error message
 */

void terminate(
	int id,		/* IN - process rank */
	char *errmsg)	/* IN - message to print */	
{
	if (!id) {
		printf("Error: %s\n", errmsg);
		fflush(stdout);
	}
	MPI_Finalize();
	exit (-1);
}

void read_row_striped_matrix(
	char		  *s,		/* IN - file name */
	void		***subs,	/* OUT - 2d submatrix indices */
	void		 **storage,	/* OUT - submatrix */
	MPI_Datatype	   dtype,	/* IN - matrix ele type */
	int		  *m,		/* OUT - matrix rows */
	int		  *n,		/* OUT - matrix columns */
	MPI_Comm	   comm)	/* IN - communicator */
{
	int	datum_size;	/* size of matrix element */
	int	i;
	int	id;		/* process rank */
	FILE	*infileptr;	/* input file pointer */
	int	local_rows;	/* rows of this process */
	void	**lptr;		/* pointer into 'subs' */
	int	p;		/* number of processes */
	void	*rptr;		/* pointer into 'storage' */
	MPI_Status status;	/* result of recieve */
	int	x;		/* result of read */

	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &id);
	datum_size = get_size(dtype);

	/* process p-1 opens file, reads size of matrix and 
         * broadcasts matrix dimensions to other processes 
	 */
	
	if (id == p-1) {
		infileptr = fopen(s, "r");
		if (infileptr == NULL) *m = 0;
		else {
			fread (m, sizeof(int), 1, infileptr);
			fread (n, sizeof(int), 1, infileptr);
		}
	}
	MPI_Bcast(m, 1, MPI_INT, p-1, comm);

	if (!(*m)) MPI_Abort(MPI_COMM_WORLD, OPEN_FILE_ERROR);

	MPI_Bcast(n, 1, MPI_INT, p-1, comm);

	local_rows = BLOCK_SIZE(id, p, *m);
	/* local_rows >= BLOCK_SIZE(id, p, *m) */
	
	*storage = (void*)my_malloc(id, local_rows * *n * datum_size);
	*subs = (void**)my_malloc(id, local_rows*PTR_SIZE);

	lptr = (void*) &(*subs[0]);
	rptr = (void*) *storage;
	for (i = 0; i < local_rows; i++) {
		*(lptr++) = (void*) rptr;
		rptr += *n * datum_size;
	}

	/* process p-1 reads blocks of rows from file and 
	 * sends each block to the correct destination process
	 * the last block it keeps         
         */

	if (id == p-1) {
		for (i = 0; i < p-1; i++) {
			x = fread(*storage, datum_size, 
				BLOCK_SIZE(i, p, *m) * *n, infileptr);
			MPI_Send(*storage, 
				BLOCK_SIZE(i, p, *m) * *n, dtype, 
				i, DATA_MSG, comm);
		}
		x = fread(*storage, datum_size, 
			local_rows * *n, infileptr);
		fclose(infileptr);
	}
	else {
		MPI_Recv(*storage, local_rows * *n, dtype, 
			p-1, DATA_MSG, comm, &status);
	}
}

int write_random_adjacency_matrix(char *filename, MPI_Datatype dtype, int m, int n) {

	FILE *outfileptr;
	int *buff;
	int datum_size, i, j;

	if (m < 1 || n < 1) {
		printf("Dimensions must be positive\n");
		return -1;
	}

	outfileptr = fopen (filename, "w");
	if (outfileptr == NULL) {
		printf("Cannot open %s for writing\n", filename);
		return -1;
	}

	fwrite (&m, sizeof(int), 1, outfileptr);
	fwrite (&n, sizeof(int), 1, outfileptr);

	datum_size = get_size(dtype);

	buff = (int*)malloc (n*datum_size);
	if (buff == NULL) {
		fclose (outfileptr);
		return -1;
	}

	for (j = 0; j < m; j++) {
		for (i = 0; i < n; i++) {
			buff[i] = (int)(rand() % 20);
			if (buff[i] < 5) {
				buff[i] = (int)RAND_MAX;
			}
			else {
				buff[i] = (int)buff[i] - 5;
			}
		}
		buff[j] = 0;
		fwrite (buff, datum_size, n, outfileptr);
	}

	free (buff);
	fclose (outfileptr);
	return 0;
}

int write_random_matrix(char *filename, MPI_Datatype dtype, int m, int n) {

	FILE *outfileptr;
	double *buff;
	int datum_size, i, j;

	if (m < 1 || n < 1) {
		printf("Dimensions must be positive\n");
		return -1;
	}

	outfileptr = fopen (filename, "w");
	if (outfileptr == NULL) {
		printf("Cannot open %s for writing\n", filename);
		return -1;
	}

	fwrite (&m, sizeof(int), 1, outfileptr);
	fwrite (&n, sizeof(int), 1, outfileptr);

	datum_size = get_size(dtype);

	buff = (double*)malloc (n*datum_size);
	if (buff == NULL) {
		fclose (outfileptr);
		return -1;
	}

	for (j = 0; j < m; j++) {
		for (i = 0; i < n; i++) {
			buff[i] = (int)(rand() % 20) - 10;
			buff[i] = buff[i]/10;
		}
		fwrite (buff, datum_size, n, outfileptr);
	}

	free (buff);
	fclose (outfileptr);
	return 0;
}

int write_random_vector(char *filename, MPI_Datatype dtype, int n) {

	FILE *outfileptr;
	double *buff;
	int datum_size, i, j;

	if (n < 1) {
		printf("Dimensions must be positive\n");
		return -1;
	}

	outfileptr = fopen (filename, "w");
	if (outfileptr == NULL) {
		printf("Cannot open %s for writing\n", filename);
		return -1;
	}

	fwrite (&n, sizeof(int), 1, outfileptr);

	datum_size = get_size(dtype);

	buff = (double*)malloc (n*datum_size);
	if (buff == NULL) {
		fclose (outfileptr);
		return -1;
	}

	for (i = 0; i < n; i++) {
		buff[i] = (int)(rand() % 20) - 10;
		buff[i] = buff[i]/10;
	}
	fwrite (buff, datum_size, n, outfileptr);

	free (buff);
	fclose (outfileptr);
	return 0;
}

int write_random_binary_matrix(char *filename, MPI_Datatype dtype, int m, int n) {

	FILE *outfileptr;
	int *buff;
	int datum_size, i, j, seconds;

	if (m < 1 || n < 1) {
		printf("Dimensions must be positive\n");
		return -1;
	}

	outfileptr = fopen (filename, "w");
	if (outfileptr == NULL) {
		printf("Cannot open %s for writing\n", filename);
		return -1;
	}

	fwrite (&m, sizeof(int), 1, outfileptr);
	fwrite (&n, sizeof(int), 1, outfileptr);

	datum_size = get_size(dtype);

	buff = (int*)malloc (n*datum_size);
	if (buff == NULL) {
		fclose (outfileptr);
		return -1;
	}

	seconds = time(NULL);
	srand((unsigned int)seconds);
	for (j = 0; j < m; j++) {
		for (i = 0; i < n; i++) {
			buff[i] = (int)(rand() % 4);
			if (buff[i] > 0) {
				buff[i] = 0;
			}
			else {
				buff[i] = 1;
			}
		}
		fwrite (buff, datum_size, n, outfileptr);
	}

	free (buff);
	fclose (outfileptr);
	return 0;
}

int write_buffered_matrix(char *filename, MPI_Datatype dtype, void *buff, int m, int n) {
	
	FILE *outfileptr;
	int datum_size;

	if (m < 1 || n < 1) {
		printf("Dimensions must be positive\n");
		return -1;
	}

	outfileptr = fopen (filename, "w");
	if (outfileptr == NULL) {
		printf("Cannot open %s for writing\n", filename);
		return -1;
	}

	fwrite (&m, sizeof(int), 1, outfileptr);
	fwrite (&n, sizeof(int), 1, outfileptr);

	datum_size = get_size(dtype);
	fwrite (buff, datum_size, m*n, outfileptr);
	fclose (outfileptr);
	return 0;
}

void print_submatrix(
	void		**a,	/* IN - doubly subscripted array */
	MPI_Datatype	dtype,	/* IN - type of array elements */
	int		rows,	/* IN - matrix rows */
	int		cols)	/* IN - matrix columns */
{
	int i, j, el;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			if (dtype == MPI_DOUBLE)
				printf("%6.3f", ((double**)a)[i][j]);
			else if (dtype == MPI_FLOAT)
				printf("%6.3f", ((float**)a)[i][j]);
			else if (dtype == MPI_INT) {
				el = ((int**)a)[i][j];
				if (el > 1000000)
					printf("   Inf");
				else 
					printf("%6d", el);
			}
		}
		putchar('\n');
	}
}

void print_binary_submatrix(
	void		**a,	/* IN - doubly subscripted array */
	MPI_Datatype	dtype,	/* IN - type of array elements */
	int		rows,	/* IN - matrix rows */
	int		cols)	/* IN - matrix columns */
{
	int i, j, el;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			if (dtype == MPI_DOUBLE) {
				if (((double**)a)[i][j] > 0)
					printf("o");
				else
					printf(" ");
			} else if (dtype == MPI_FLOAT) {
				if (((float**)a)[i][j] > 0)
					printf("o");
				else
					printf(" ");
			} else if (dtype == MPI_INT) {
				if (((int**)a)[i][j] > 0)
					printf("o");
				else
					printf(" ");
			}
		}
		putchar('\n');
	}
}

void print_row_striped_matrix(
	void **a,		/* IN - 2D array */
	MPI_Datatype dtype,	/* IN - matrix element type */
	int m,			/* IN - matrix rows */
	int n,			/* IN - matrix columns */
	MPI_Comm comm)		/* IN - communicator */
{
	MPI_Status	status;
	void		*bstorage;
	void		**b;
	int		datum_size;
	int		i;
	int		id;
	int		local_rows;
	int		max_block_size;
	int		prompt;
	int		p;

	MPI_Comm_rank(comm, &id);
	MPI_Comm_size(comm, &p);
	local_rows = BLOCK_SIZE(id,p,m);
	
	if (!id) {
		print_submatrix(a, dtype, local_rows, n);
		if (p > 1) {
			datum_size = get_size(dtype);
			max_block_size = BLOCK_SIZE(p-1,p,m);
			bstorage = my_malloc(id, max_block_size*n*datum_size);
			b = (void**)my_malloc(id, max_block_size*sizeof(void*));
			b[0] = bstorage;
			for (i = 1; i < max_block_size; i++) {
				b[i] = b[i-1] + n*datum_size;
			}
			for (i = 1; i < p; i++) {
				MPI_Send(&prompt, 1, MPI_INT, i, PROMPT_MSG,
					MPI_COMM_WORLD);
				MPI_Recv(bstorage, BLOCK_SIZE(i,p,m)*n, dtype, 
					i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
				print_submatrix(b, dtype, BLOCK_SIZE(i,p,m), n);
			}
			free(b);
			free(bstorage);
		}
		putchar('\n');
	}
	else {
		MPI_Recv(&prompt, 1, MPI_INT, 0, PROMPT_MSG, 
			MPI_COMM_WORLD, &status);
		MPI_Send(*a, local_rows*n, dtype, 0, RESPONSE_MSG,
			MPI_COMM_WORLD);
	}
}

void print_row_striped_binary_matrix(
	void **a,		/* IN - 2D array */
	MPI_Datatype dtype,	/* IN - matrix element type */
	int m,			/* IN - matrix rows */
	int n,			/* IN - matrix columns */
	MPI_Comm comm)		/* IN - communicator */
{
	MPI_Status	status;
	void		*bstorage;
	void		**b;
	int		datum_size;
	int		i;
	int		id;
	int		local_rows;
	int		max_block_size;
	int		prompt;
	int		p;

	MPI_Comm_rank(comm, &id);
	MPI_Comm_size(comm, &p);
	local_rows = BLOCK_SIZE(id,p,m);
	
	if (!id) {
		for (i = 0; i < n; i++) {
			printf("-");
		}
		putchar('\n');
		print_binary_submatrix(a, dtype, local_rows, n);
		if (p > 1) {
			datum_size = get_size(dtype);
			max_block_size = BLOCK_SIZE(p-1,p,m);
			bstorage = my_malloc(id, max_block_size*n*datum_size);
			b = (void**)my_malloc(id, max_block_size*sizeof(void*));
			b[0] = bstorage;
			for (i = 1; i < max_block_size; i++) {
				b[i] = b[i-1] + n*datum_size;
			}
			for (i = 1; i < p; i++) {
				MPI_Send(&prompt, 1, MPI_INT, i, PROMPT_MSG,
					MPI_COMM_WORLD);
				MPI_Recv(bstorage, BLOCK_SIZE(i,p,m)*n, dtype, 
					i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
				print_binary_submatrix(b, dtype, BLOCK_SIZE(i,p,m), n);
			}
			free(b);
			free(bstorage);
		}
		putchar('\n');
		for (i = 0; i < n; i++) {
			printf("-");
		}
		putchar('\n');
	}
	else {
		MPI_Recv(&prompt, 1, MPI_INT, 0, PROMPT_MSG, 
			MPI_COMM_WORLD, &status);
		MPI_Send(*a, local_rows*n, dtype, 0, RESPONSE_MSG,
			MPI_COMM_WORLD);
	}
}

void read_replicated_vector(
	char *a,		/* IN - file name */
	void **v,		/* OUT - vector */
	MPI_Datatype dtype,	/* IN - vector type */
	int *n,			/* OUT - vector length */
	MPI_Comm comm)		/* IN - communicator */
{
	int datum_size;
	int i, id;
	FILE *infileptr;
	int p;

	MPI_Comm_rank (comm, &id);
	MPI_Comm_size (comm, &p);
	datum_size = get_size(dtype);

	if (id == (p-1)) {
		infileptr = fopen(a, "r");
		if (infileptr == NULL) {
			*n = 0;
		}
		else {
			fread (n, sizeof(int), 1, infileptr);
		}
	}
	MPI_Bcast (n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	
	if (!*n) terminate(id, "Cannot open vector file");
	*v = my_malloc (id, *n * datum_size);
	
	if (id == (p-1)) {
		fread (*v, datum_size, *n, infileptr);
		fclose (infileptr);
	}
	// all processes get a copy of the vector
	MPI_Bcast (*v, *n, dtype, p-1, MPI_COMM_WORLD);
}

void print_subvector(
	char *a,		/* IN - array pointer */
	MPI_Datatype dtype,	/* IN - vector type */
	int n)			/* IN - vector length */
{
	int i;
	for (i = 0; i < n; i++) {
		if (dtype== MPI_DOUBLE) {
			printf ("%6.3f ", ((double*)a)[i]);
		}
		else if (dtype == MPI_FLOAT) {
			printf ("%6.3f ", ((float*)a)[i]);
		}
		else if (dtype == MPI_INT) {
			printf ("%6.3d ", ((int*)a)[i]);
		}
	}
}

void print_replicated_vector(
	void *v,		/* IN - vector */
	MPI_Datatype dtype,	/* IN - vector type */
	int  n,			/* IN vector length */
	MPI_Comm comm)		/* IN - communicator */
{
	int id;

	MPI_Comm_rank (comm, &id);
	if (!id) {
		print_subvector (v, dtype, n);
		printf ("\n\n");
	}
}

void create_mixed_xfer_arrays(
	int id,
	int p,
	int n,
	int **count,	/* OUT - arrays of counts */
	int **disp)	/* OUT - array of displacements */
{
	int i;

	*count = my_malloc (id, p*sizeof(int));
	*disp  = my_malloc (id, p*sizeof(int));
	(*count)[0] = BLOCK_SIZE(0, p, n);
	(*disp)[0] = 0;
	for (i = 1; i < p; i++) {
		(*disp)[i] = (*disp)[i-1] + (*count)[i-1];
		(*count)[i] = BLOCK_SIZE(i,p,n);
	}
}

void replicate_block_vector (
	void *ablock,		/* IN - block-distributed vector */
	int n, 			/* IN - elements in vector */
	void *arep,		/* OUT - replicated vector */
	MPI_Datatype dtype,	/* IN */
	MPI_Comm comm)		/* IN */
{
	int *cnt;	/* elelemnts contributed by each process */
	int *disp;	/* displacement in concatenated array */
	int id, p;

	MPI_Comm_rank (comm, &id);
	MPI_Comm_size (comm, &p);
	
	create_mixed_xfer_arrays (id, p, n, &cnt, &disp);

	MPI_Allgatherv (ablock, cnt[id], dtype, arep, cnt, 
		disp, dtype, comm);

	free (cnt);
	free (disp);	
}

