#include "domain.h"
#include <mpi.h>

void set_left_boundary(float *u, long long int rpb, long long int cpb)
{
        int i;
        for(i = 0; i < cpb * rpb; i++)
        {
                if (i % cpb == 0)
                {
                        u[i] = 0;
                }
        }
}

void set_right_boundary(float *u, long long int rpb, long long int cpb)
{
        int i;
        for( i = 0; i < cpb * rpb; i++)
        {
                if ((i+1)  % cpb == 0)
                {
                        u[i] = 0;
                }
        }
}

void set_top_boundary(float *u, long long int rpb, long long int cpb)
{
        int i;
        for( i = 0; i < cpb; i++)
        {
                u[i] = 0;
        }

}

void set_bot_boundary(float *u, long long int rpb, long long int cpb)
{
        int i;
        for(i = cpb * (rpb - 1); i < cpb * rpb; i++)
        {
                u[i] = 0;
        }
}

void update_u(float *u, float *new_val, long long int rpb, long long int cpb)
{
	int i, k;
	for (k = 0; k < rpb; k++)
        {
    		for (i = 0; i < cpb; i++)
                {
                	u[i+ k * cpb] = new_val[i + k * cpb];
                }
         }
}



void print_domain(float *u, float *dum, int id, long long int cpb, long long int rpb, int nprocs, MPI_Comm comm)
{

        MPI_Gather(&u[0], cpb * rpb, MPI_FLOAT, &dum[0], cpb * rpb, MPI_FLOAT, 0, comm);
        if (id == 0)
        {
                int j;
                for(j = 0; j < cpb * rpb * nprocs; j++)
                {
                        if (j % cpb == 0)
                                printf("\n");
                        if (j % (cpb * rpb) == 0)
                        	printf("\n"); 
                        printf("%f\t", dum[j]);
                }
                printf("\n\n");
        }
}
