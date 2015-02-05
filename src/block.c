//**************************************************//
// Solving finite difference based poisson equation 
// 1. Assuming n X n matrix                         
// 2. n % procs = 0
// 3. Forcing parameter: f = 2.0
//**************************************************//

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include "domain.h"
#include "block.h"

int main(argc, argv)
int argc;
char *argv[];
{
	int ierr;
	int nprocs, id;
	
	
	MPI_Comm comm = MPI_COMM_WORLD;
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(comm, &nprocs);
	ierr = MPI_Comm_rank(comm, &id);

	double tstart, tstop, t;
	long long int n_node;
	int sqp;
	sqp = sqrt(nprocs);
	float h;


	if(id == 0)
	{
		printf("\nEnter the number of nodes\t:\t ");
		scanf("%lld", &n_node);
		if ((n_node) % sqp != 0)
		{
			printf("Cant divide blocks\n");
			return(0);
		}
	}
	
	MPI_Bcast(&n_node, 1, MPI_LONG_LONG_INT, 0, comm);
	h = 1.0/(n_node - 1.0);
	long long int rpb = n_node/sqp; //rows per block
	long long int cpb = n_node/sqp; // columns per block
	
	float *u, *new_val, *top_buffer, *bot_buffer, *left_buffer, *right_buffer;
	u = malloc((rpb * cpb) * sizeof(float));
	new_val = malloc((rpb * cpb) * sizeof(float));
	top_buffer = malloc(cpb * sizeof(float));
	bot_buffer = malloc(cpb * sizeof(float));
	left_buffer = malloc(rpb * sizeof(float));
	right_buffer = malloc(rpb * sizeof(float));
	
	int j, s;
	float *dum;
	if (id == 0)
	{
		dum = malloc(n_node * n_node * sizeof(float));
	}
	float f = 2.0;

	//Initial guess
	for (j = 0; j < rpb * cpb ; j++)
	{
		u[j] = 3.0;
	}
	
	set_boundaries(&u[0], rpb, cpb, nprocs, id);

	for(s = 0; s < cpb; s++)
	{
		top_buffer[s] = 0;
		bot_buffer[s] = 0;
		left_buffer[s] = 0;
		right_buffer[s] = 0;
	}	

	int pol = 0;
	tstart = MPI_Wtime();
	while(pol < 1)
	{ 
		if(nprocs != 1)
		{
			send_to_left(&u[0], &left_buffer[0], rpb, cpb, nprocs, id, comm);

			send_to_right(&u[0], &right_buffer[0], rpb, cpb, nprocs, id, comm);

			send_to_bot(&u[0], &bot_buffer[0], rpb, cpb, nprocs, id, comm);

			send_to_top(&u[0], &top_buffer[0], rpb, cpb, nprocs, id, comm);
		}

		check_neighbor(&u[0], &top_buffer[0], &bot_buffer[0], &right_buffer[0], &left_buffer[0], cpb, nprocs, id, rpb, &new_val[0], f, h);
	
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
		printf("nprocs: %d n_nodes: %lld Max time: %f", nprocs, n_node, a);
	}
	MPI_Finalize();
	return(0); 
}

//Set the boundary conditions
void set_boundaries(float *u, long long int rpb, long long int cpb, int nprocs, int id)
{
	int sqp;
	sqp = sqrt(nprocs);
	//Set the boundaries
	if (id < sqp)
	{
		set_top_boundary(&u[0], rpb, cpb);
	} 
	if(id % sqp == 0)
	{
		set_left_boundary(&u[0], rpb, cpb);
	}
	if((id + 1) % sqp == 0)
	{
		set_right_boundary(&u[0], rpb, cpb);
	}
	if(id >= sqp*(sqp-1))
	{
		set_bot_boundary(&u[0], rpb, cpb);
	}
}


//send data to top buffer
void send_to_top(float *u, float *top_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm)
{
	int src, dest, sqp;
	sqp = sqrt(nprocs);
	if (id >= sqp * (sqp-1))
	{
		src = id - sqp;
		dest = MPI_PROC_NULL;
	}
	else if (id < sqp)
	{
		src = MPI_PROC_NULL;
		dest = id + sqp;
	}
	else
	{
		src = id - sqp;
		dest = id + sqp; 
	}
	MPI_Sendrecv(&u[(rpb -1) * cpb], cpb, MPI_FLOAT, dest,0, &top_buffer[0], cpb, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
}


//send data to bottom buffer
void send_to_bot(float *u, float *bot_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm)
{
	int src, dest, sqp;
	sqp = sqrt(nprocs);
	if (id < sqp)
	{
		dest = MPI_PROC_NULL;
		src = id + sqp;
	}
	else if (id >= sqp * (sqp -1))
	{
		dest = id - sqp;
		src = MPI_PROC_NULL;
	}
	else	
	{
		dest = id -sqp;
		src = id + sqp;
	}
	MPI_Sendrecv(&u[0], cpb, MPI_FLOAT, dest, 0, &bot_buffer[0], cpb, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
}


//send data to right buffer
void send_to_right(float *u, float *right_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm)
{

	int src, dest, sqp;
	sqp = sqrt(nprocs);
	
	MPI_Datatype newvector;
	MPI_Type_vector(rpb, 1, cpb, MPI_FLOAT, &newvector);
	MPI_Type_commit(&newvector);
	
	if (id % sqp == 0)
	{
		dest = MPI_PROC_NULL;
		src = id + 1;
	}
	else if ((id + 1) % sqp == 0)
	{
		dest = id - 1;
		src = MPI_PROC_NULL;
	}
	else	
	{
		dest = id - 1;
		src = id + 1;
	}
	MPI_Sendrecv(&u[0], 1, newvector, dest, 0, &right_buffer[0], rpb, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
	MPI_Type_free(&newvector);
}


//send data to left buffer
void send_to_left(float *u, float *left_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm)
{
	int src, dest, sqp;
	sqp = sqrt(nprocs);
	MPI_Datatype newvector;
	MPI_Type_vector(rpb, 1, cpb, MPI_FLOAT, &newvector);
	MPI_Type_commit(&newvector);
	
	if (id % sqp == 0)
	{
		src = MPI_PROC_NULL;
		dest = id + 1;
	}
	else if ((id + 1) % sqp == 0)
	{
		src = id - 1;
		dest = MPI_PROC_NULL;
	}
	else	
	{
		dest = id + 1;
		src = id - 1;
	}
	MPI_Sendrecv(&u[cpb - 1], 1, newvector, dest, 0, &left_buffer[0], rpb, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);
	MPI_Type_free(&newvector);
}

void check_neighbor(float *u, float *top_buffer, float *bot_buffer, float *right_buffer, float *left_buffer, long long int cpb, int nprocs, int id, long long int rpb, float *new_val, float f, float h)
{
	int i, k, sqp;
	sqp = sqrt(nprocs);
	float left_val, right_val, top_val, bottom_val; 
	for (k = 0; k < rpb; k++)
	{
		for (i = 0; i < cpb; i++)
		{

			//check left neighbors
			if (i % cpb == 0)
			{
				if (id % sqp == 0)
				{
					left_val = 0.;
				}
				else
				{
					left_val = left_buffer[k];
				}
			}
			else
			{
				left_val = u[k * cpb + i - 1];
			}
	
			//check right neighbors
			if ((i+1) % cpb == 0)
			{
				if ((id + 1)%sqp == 0)
				{
					right_val = 0.;
				}
				else
				{
					right_val = right_buffer[k];
				}
			}
			else
			{
				right_val = u[k * cpb + i + 1];
			}

		
			//check top neighbors
			if (k == 0)
			{
				if (id < sqp)
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

			//check bottom neighbors
			if (k == rpb - 1)
			{
				if (id >  sqp * (sqp -1))
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
