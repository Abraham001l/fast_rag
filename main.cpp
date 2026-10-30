#include <cstdio>
#include <mpi.h>
#include "doc_utilities.h"
#include "embedding_utilities.h"
#include "graph_utilities.h"
using namespace std;

constexpr size_t one_GB = 1024 * 1024 * 1024;

int main(int argc, char** argv) {
    // initial MPI setup
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // creating 1GB shared memory
    MPI_Comm shm_comm;
    MPI_Comm_split_type(
        MPI_COMM_WORLD, 
        MPI_COMM_TYPE_SHARED, 
        0, 
        MPI_INFO_NULL, 
        &shm_comm);

    MPI_Win shm_window;
    void* shm_data_ptr = nullptr;

    MPI_Aint shm_size = (rank == 0) ? sizeof(char) * one_GB : 0;

    MPI_Win_allocate_shared(
        shm_size, 
        sizeof(char), 
        MPI_INFO_NULL, 
        shm_comm, 
        &shm_data_ptr, 
        &shm_window);

    // getting pointer to rank 0 shm
    MPI_Aint shm_tot_bytes;
    int displacement_unit;
    char* shm_base_ptr;

    MPI_Win_shared_query(
        shm_window, 
        0, 
        &shm_tot_bytes, 
        &displacement_unit, 
        &shm_base_ptr);
    
    // having rank 0 fill in shm and assigning rank offsets
    MPI_Win_lock_all(0, shm_window);
    
    // skeleton struct for docs
    docs_container d = {sizeof(size_t)*3, 0, 0};

    if (rank == 0) {
        vector<string> doc_paths = {"test_text_files/doc1.txt", "test_text_files/doc2.txt", "test_text_files/doc3.txt", "test_text_files/doc4.txt",
        "test_text_files/doc5.txt", "test_text_files/doc6.txt", "test_text_files/doc7.txt", "test_text_files/doc8.txt", "test_text_files/doc9.txt",
        "test_text_files/doc10.txt"};

        populate_docs_container(d, shm_base_ptr, doc_paths);
        calc_rank_offsets(d, shm_base_ptr, world_size);
    }

    // synchronizing shm for processes
    MPI_Win_sync(shm_window);
    MPI_Barrier(shm_comm);
    MPI_Win_sync(shm_window);

    // having other ranks define skeleton struct for docs
    if (rank != 0) {
        interpret_filled_docs_container(d, shm_base_ptr, world_size);
    }

    // creating & filling work containers
    work_container w = {0};
    fill_work_container(w, d, shm_base_ptr, rank);

    // chunking the work memory into lines
    std::vector<std::string_view> chunks;
    chunk_memory(w, chunks);

    // embedding the chunks
    vector<float> embeddings;
    embed_chunks(chunks, embeddings);

    // building adjacency matrix from embeddings
    matrix_rep_adjacency adj;
    build_adjacency(adj, embeddings, 0.8f); // Initial threshold; tune for your documents.

    // building CSR graph from adjacency matrix
    csr_graph csr;
    build_csr(csr, adj);

    // printing the CSR graph for verification
    for (size_t i = 0; i < csr.offsets.size() - 1; ++i) {
        printf("Node %zu: ", i);
        for (size_t j = csr.offsets[i]; j < csr.offsets[i + 1]; ++j) {
            printf("%zu ", csr.neighbors[j]);
        }
        printf("\n");
    }

    MPI_Win_unlock_all(shm_window);
    MPI_Win_free(&shm_window);
    MPI_Comm_free(&shm_comm);
    MPI_Finalize();
}
