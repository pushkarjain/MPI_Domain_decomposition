#include <mpi.h>

void set_boundaries(float *u, long long int rpb, long long int cpb, int nprocs, int id);
void send_to_right(float *u, float *right_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm);
void send_to_left(float *u, float *left_buffer, long long int rpb, long long int cpb, int nprocs, int id, MPI_Comm comm);
void send_to_top(float *u, float *top_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm);
void send_to_bot(float *u, float *bot_buffer, long long int rpb, long long int n_node, int nprocs, int id, MPI_Comm comm);
void check_neighbor(float *u, float *top_buffer, float *bot_buffer, float *right_buffer, float *left_buffer, long long int cpb, int nprocs, int id, long long int rpb, float *new_val, float f, float h);
