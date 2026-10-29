#pragma once
#include <vector>
#include <fstream>
#include <span>
#include <algorithm>
#include <cstring>
#include <stdexcept>
using namespace std;

struct docs_container {
    size_t doc_mem_offset;
    size_t doc_sizes_offset;
    size_t rank_offsets_offset;
    span<char> doc_mem;
    span<size_t> doc_sizes;
    span<size_t> rank_offsets;
};

struct work_container {
    size_t work_mem_size;
    span<char> work_mem;
};

void populate_docs_container(docs_container& d, char* shm_data_ptr, const vector<string>& doc_paths);

void calc_rank_offsets(docs_container& d, char* shm_data_ptr, size_t world_size);

void interpret_filled_docs_container(docs_container& d, char* shm_data_ptr, size_t world_size);

void fill_work_container(work_container& w, docs_container& d, char* shm_data_ptr, int rank);