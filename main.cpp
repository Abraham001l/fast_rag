#include <cstdio>
#include <mpi.h>
using namespace std;

int main(int argc, char** argv) {
    // initial MPI setup
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int shm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &shm_rank);

    // ensuring we have min num cores assgined for parallel execution
    if (world_size < 3) {
        if (shm_rank == 0) {
            printf("Error: Need -np arguement to be atleast 3 for parallel execution\n");
        }
        MPI_Finalize();
        return 1;
    }

    

    





    return 0;
}