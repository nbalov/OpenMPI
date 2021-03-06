#include <mpi.h>
#include <stdio.h>

#define INTERVALS 1000000

double pi_segment(double a, double b, int n);

int main(int argc, char *argv[]) {
	int i  = 0;
	int id = 0;
	int p  = 0;
	
	double a = 0;
	double delta = 0;
	int nseg = 1;

	double fsum = 0;
	double global_fsum = 0;
	double time = 0;
	double global_time = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	MPI_Barrier(MPI_COMM_WORLD);
	time = -MPI_Wtime();

	delta = (double)1 / (double)p;
	nseg = INTERVALS/p;
	a = id*delta;

	if (id < p-1) {
		fsum = pi_segment(a, a + delta, nseg);
	}
	else {
		fsum = pi_segment(a, 1, INTERVALS-(p-1)*nseg);
	}

	MPI_Reduce(&fsum, &global_fsum, 1, MPI_DOUBLE,
		MPI_SUM, 0, MPI_COMM_WORLD);

	time += MPI_Wtime();

	MPI_Reduce(&time, &global_time, 1, MPI_DOUBLE,
		MPI_SUM, 0, MPI_COMM_WORLD);

	printf("Process %d is done\n", id);
	fflush(stdout);

	MPI_Finalize();

	if (id == 0) {
		printf("Approximate PI: %10.8f\n", global_fsum);
		printf("Time elapsed: %f sec\n", global_time);
		fflush(stdout);
	}

	return 0;
}

double pi_segment(double a, double b, int n) {
	int i = 0;
	double delta = 0;
	double fsum = 0;
	double x = 0;

	if (a<0 || b>1 || a>=b || n<1) {
		return 0;
	}

	delta = (b-a)/(double)n;
	x = a + 0.5*delta;
	for (i = 0; i < n; i++) {
		fsum += 4.0/(1.0+x*x);
		x += delta;
	}

	return (fsum*delta);
}

