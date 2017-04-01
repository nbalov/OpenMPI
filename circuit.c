/*
 * Circuit Satisfiability
 * J. Quinn, 3 Sep 2002
 */

#include <mpi.h>
#include <stdio.h>

int check_circuit(int, int);

int main(int argc, char *argv[]) {
	int i  = 0;
	int id = 0;	/* Process rank */
	int p  = 0;	/* Number of precesses */
	int solutions = 0;	/* Solutions found by this process */
	int global_solutions = 0;
	double elapsed_time = 0;
	double global_elapsed_time = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();

	for (i = id; i < (1<<16); i += p) {
		solutions += check_circuit(id, i);
	}
	
	MPI_Reduce(&solutions, &global_solutions, 1, 
		MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	elapsed_time += MPI_Wtime();

	MPI_Reduce(&elapsed_time, &global_elapsed_time, 1, 
		MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	printf("Process %d is done in %f sec\n", id, elapsed_time);
	fflush(stdout);

	MPI_Finalize();
	
	if (id == 0) {
		printf("There are %d different solutions\n", 
			global_solutions);
		printf("Total time is %f sec\n", 
			global_elapsed_time);
	}

	return 0;
}

#define EXTRACT_BIT(n, i) ((1<<i) & n ? 1 : 0)

int check_circuit(int id, int z) {

	int v[16];
	int i = 0;
	int checks = 0;

	for (i = 0; i < 16; i++) {
		v[i] = EXTRACT_BIT(z, i);
	}

	if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
		&& (!v[3]  || !v[4])  && ( v[4]  || !v[5]) 
		&& ( v[5]  || !v[6])  && ( v[5]  ||  v[6]) 
		&& ( v[6]  || !v[15]) && ( v[7]  || !v[8]) 
		&& (!v[7]  || !v[13]) && ( v[8]  ||  v[9]) 
		&& ( v[8]  || !v[9])  && (!v[9]  || !v[10]) 
		&& ( v[9]  ||  v[11]) && ( v[10] ||  v[11]) 
		&& ( v[12] ||  v[13]) && ( v[13] || !v[14]) 
		&& ( v[14] ||  v[15])) {
		checks++;
		printf("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id, 
			v[0], v[1], v[3], v[4], v[5], v[6], v[7], 
			v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
		fflush(stdout);
	}

	return checks;
}

