#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include "domain.h"

void set_boundaries(float *u, long long int rpb, long long int cpb, int nprocs, int id);
void send_to_top(float *u, float *top_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm);
void send_to_bot(float *u, float *bot_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm);
void check_neighbor(float *u, float *top_buffer, float *bot_buffer, long long int cpb, int nprocs, int id, long long int rpb, float *new_val, float f, float h);

int main(argc, argv)
int argc;
char *argv[];
{

// 1. Assuming n X n matrix
// 2. n % procs = 0

	int ierr;
	int nprocs, id;

	MPI_Comm comm = MPI_COMM_WORLD;
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(comm, &nprocs);
	ierr = MPI_Comm_rank(comm, &id);

	long long int n_node;
	float h;

// Solving finite difference based poisson equation

	if(id == 0)
	{
		printf("\nEnter the number of nodes\t:\t ");
		scanf("%lld", &n_node);
		if ((n_node)%nprocs != 0)
		{
			printf("Cant divide rows\n");
			return(0);
		}
	}
	
	MPI_Bcast(&n_node, 1,MPI_LONG_LONG_INT, 0, comm);
	h = 1.0/(n_node -1.0);
	long long int rpb = n_node/nprocs; //rows per block
	long long int cpb = n_node; //rows per block
	
	float *u, *new_val, *top_buffer, *bot_buffer;
	u = malloc((rpb * cpb) * sizeof(float));
	new_val = malloc((rpb * cpb) * sizeof(float));
	top_buffer = malloc(cpb * sizeof(float));
	bot_buffer = malloc(cpb * sizeof(float));
	
	int j, s;
	float *dum;
	if (id == 0)
	{
		dum = malloc(n_node * n_node * sizeof(float));
	}
	float f = 2.0;
	double tstart, tstop, t;

	//Initial guess
	for (j = 0; j < cpb * rpb ; j++)
	{
		u[j] = 3.0;
	}
	
	set_boundaries(&u[0], rpb, cpb, nprocs, id);

	//print_domain(&u[0], &dum[0], id, cpb, rpb, nprocs, comm);
	
	for(s = 0; s < cpb; s++)
	{
		top_buffer[s] = 0;
		bot_buffer[s] = 0;
	}	

	int pol = 0;
	tstart = MPI_Wtime();	
	
	while(pol < 1)
	{	
		if(nprocs != 1)
		{ 
			send_to_bot(&u[0], &bot_buffer[0], rpb, n_node, nprocs, id, comm);

			send_to_top(&u[0], &top_buffer[0], rpb, n_node, nprocs, id, comm);
		}
		
		check_neighbor(&u[0], &top_buffer[0], &bot_buffer[0], cpb, nprocs, id, rpb, &new_val[0], f, h);

		update_u(&u[0], &new_val[0], rpb, cpb);

		pol = pol + 1;
	}

	print_domain(&u[0], &dum[0], id, cpb, rpb, nprocs, comm);
	tstop = MPI_Wtime();	
	t = tstop - tstart;
	double a;
	MPI_Reduce(&t, &a, 1, MPI_DOUBLE, MPI_MAX, 0, comm);		
	if (id == 0)
	{
		FILE *f = fopen("outfile.txt", "w");
		fprintf(f, "nprocs: %d, n_node: %ld, pol: %d, Max time: %f\n", nprocs, n_node, pol, a);
		fclose(f);
		printf("nprocs: %d, n_node: %ld, pol: %d, Max time: %f\n", nprocs, n_node, pol, a);
		printf("Max time: %f\n", a);
	}
	MPI_Finalize();
	return(0);
}


void set_boundaries(float *u, long long int rpb, long long int cpb, int nprocs, int id)
{
	//Set the boundaries
	
	if (id == 0)
	{
		set_top_boundary(&u[0], rpb, cpb);
	}
	
		set_left_boundary(&u[0], rpb, cpb);
		set_right_boundary(&u[0], rpb, cpb);

	if (id == nprocs -1)
	{
		set_bot_boundary(&u[0], rpb, cpb);
	}
}


void send_to_top(float *u, float *top_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm)
{
	int src, dest;
	//send data to top buffer
	if (id == nprocs -1)
	{
		src = id -1;
		dest = MPI_PROC_NULL;
	}
	else if (id == 0)
	{
		src = MPI_PROC_NULL;
		dest = id + 1;
	}
	else
	{
		src = id - 1;
		dest = id + 1; 
	}
	MPI_Sendrecv(&u[(rpb -1)*n_node], n_node, MPI_FLOAT, dest,0, &top_buffer[0], n_node, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
}


void send_to_bot(float *u, float *bot_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm)
{
	int src, dest;
	if (id == 0)
	{
		dest = MPI_PROC_NULL;
		src = id + 1;
	}
	else if (id == nprocs -1)
	{
		dest = id - 1;
		src = MPI_PROC_NULL;
	}
	else	
	{
		dest = id -1;
		src = id + 1;
	}
	MPI_Sendrecv(&u[0], n_node, MPI_FLOAT, dest, 0, &bot_buffer[0], n_node, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
}

void check_neighbor(float *u, float *top_buffer, float *bot_buffer, long long int cpb, int nprocs, int id, long long int rpb, float *new_val, float f, float h)
{
	int i, k;
	float left_val, right_val, top_val, bottom_val;
	for (k = 0; k < rpb; k++)
	{
		for (i = 0; i < cpb; i++)
		{
			
			//left			
			if ((i) % cpb == 0)
			{
				left_val = 0.;
			}
			else
			{
				left_val = u[k * cpb + i -1];
			}
	
			//right
	
			if ((i+1) % cpb == 0)
			{
				right_val = 0.;
			}
			else
			{
				right_val = u[k * cpb + i + 1];
			}

		
			//top
			if (k == 0)
			{
				if (id == 0)
				{
					top_val = 0.;
				}
				else
				{
					top_val = top_buffer[i];
				}
			}
			else
			{
				top_val = u[cpb * (k-1) + i];
			}

			//bottom
			if (k == rpb -1)
			{
				if (id == nprocs -1)
				{
					bottom_val = 0.;
				}
				else
				{
					bottom_val = bot_buffer[i];
				}
			}
			else
			{
				bottom_val = u[i + (cpb * (k + 1))];
			}
			new_val[i + k * cpb] = (left_val + right_val + top_val + bottom_val - h * h *f)/4.0;
		}
	}
	set_boundaries(&new_val[0], rpb, cpb, nprocs, id);
}
