#include <mpi.h>

void set_left_boundary(float *u, long long int rpb, long long int cpb);
void set_right_boundary(float *u, long long int rpb, long long int cpb);
void set_top_boundary(float *u, long long int rpb, long long int cpb);
void set_bot_boundary(float *u, long long int rpb, long long int cpb);
void update_u(float *u, float *new_val, long long int rpb, long long int cpb);
void print_domain(float *u, float *dum, int id, long long int cpb, long long int rpb, int nprocs, MPI_Comm comm);
