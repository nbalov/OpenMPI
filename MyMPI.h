/* MyMPI.h
 *
 * Programmed by Michael J. Quinn
 * 4 Sep 2002
 */

#ifndef _MyMPI_h
#define _MyMPI_h

#define DATA_MSG	0
#define PROMPT_MSG	1
#define RESPONSE_MSG	2

#define OPEN_FILE_ERROR	-1
#define MALLOC_ERROR	-2
#define TYPE_ERROR	-3

#define MIN(a,b)		((a)<(b))?(a):(b)
#define BLOCK_LOW(id,p,n)	((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n)	(BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n)	(BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j,p,n)	(((p)*((j)+1)-1)/(n))

#define PTR_SIZE	(sizeof(void*))
#define CEILING(i,j)	(((i)+(j)-1)/(j))

void terminate(int, char*);

int write_random_adjacency_matrix(char*, MPI_Datatype, int, int);
int write_random_matrix(char*, MPI_Datatype, int, int);
int write_random_vector(char*, MPI_Datatype, int);

void read_row_striped_matrix(char*, void***, void**, 
	MPI_Datatype, int*, int*, MPI_Comm);

void print_row_striped_matrix(void**, MPI_Datatype, 
	int, int, MPI_Comm);

void read_replicated_vector (char*, void**, MPI_Datatype, int*, MPI_Comm);
void print_replicated_vector (void*, MPI_Datatype, int, MPI_Comm);
void replicate_block_vector (void*, int, void*, MPI_Datatype, MPI_Comm);

#endif
