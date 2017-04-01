/*
 *     Sieve of Eratosthenes
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n) - 1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) + 1)
#define BLOCK_OWNER(index,p,n) (((p)*((index)+1)-1)/(n))

int main(int argc, char *argv[]) {

	int	count		= 0;
	double	elapsed_time	= 0;
	int	first		= 0;
	int	global_count	= 0;
	int	high_value	= 0;
	int	i		= 0;
	int	id		= 0;
	int	index		= 0;
	int	low_value	= 0;
	char	*marked 	= NULL;	/* portion of 2...`n' */
	int	n		= 0;
	int	p		= 0;
	int	proc0_size	= 0;
	int	prime		= 0;
	int	size		= 0;	/* elements in `marked' */

	MPI_Init(&argc, &argv);

	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD,  &p);

	if (argc != 2) {
		if (!id) printf("Command line: %s <m>\n", argv[0]);
		goto error;
	}

	n = atoi(argv[1]);

	/* Figure out this process' share of the array as well as 
	   the first and last array elements: 
	   reak [2,n] into p blocks
	*/
	
	low_value  = 2 + BLOCK_LOW (id,p,n-1);
	high_value = 2 + BLOCK_HIGH(id,p,n-1);
	size = BLOCK_SIZE(id,p,n-1);

	proc0_size = (n-1)/p;
	/* for the algorithm to work, all prime devisors of numbers less than `n' 
           must be in the first block, i.e. id=0 */
	if ((1+proc0_size) < (int)sqrt((double)n)) {
		if (!id) printf("Too many processes\n");
		goto error;
	}

	/* allocate this process' share of the array */
	marked = (char*) malloc (size);
	if (marked == NULL) {
		printf("Cannot allocate enough memory\n");
		goto error;
	}

	for (i = 0; i < size; i++) marked[i] = 0;

	if (!id) index = 0;
	prime = 2;
	do {
		if (prime*prime > low_value) {
			first = prime*prime - low_value;
		}
		else {
			if (!(low_value%prime)) first = 0;
			else first = prime - (low_value%prime);
		}
		for (i = first; i < size; i += prime) marked[i] = 1;

		if (!id) {
			while (marked[++index]);
			prime = index + 2;
		}

		MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);

	} while (prime*prime < n);

	count = 0;
	for (i = 0; i < size; i++) {
		if (!marked[i]) count++;
	}

	MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 
		0, MPI_COMM_WORLD);

	elapsed_time += MPI_Wtime();

	if (!id) {
		printf("%d primes are less or equal than %d\n", 
			global_count, n);
		printf("Total ellapsed time: %10.6f\n", elapsed_time);
	}

	MPI_Finalize();
	return 0;

error:
	MPI_Finalize();
	exit(1);
}

