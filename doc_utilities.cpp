#include "doc_utilities.h"

void populate_docs_container(docs_container& d, char* shm_data_ptr, const vector<string>& doc_paths) {
    vector<size_t> temp_doc_sizes;
    size_t cur_total_size = 0;

    for (const string& path : doc_paths) {
        // opening file & reading in contents
        ifstream file(path, ios::binary | ios::ate);

        if (!file.is_open()) {
            printf("Error opening file: %s\n", path.c_str());
            continue;
        }

        streamsize new_content_size = file.tellg();

        if (new_content_size <= 0) {
            printf("Error reading file: %s\n", path.c_str());
            continue;
        }

        file.seekg(0, ios::beg);
        file.read(shm_data_ptr + d.doc_mem_offset + cur_total_size, new_content_size);

        temp_doc_sizes.push_back(new_content_size);
        cur_total_size += new_content_size;
    }

    // copying in doc_size's vector
    d.doc_sizes_offset = d.doc_mem_offset + cur_total_size;
    memcpy(shm_data_ptr + d.doc_sizes_offset, temp_doc_sizes.data(), temp_doc_sizes.size() * sizeof(size_t));

    // assigning spans
    d.doc_mem = span<char>(shm_data_ptr + d.doc_mem_offset, cur_total_size);
    d.doc_sizes = span<size_t>(reinterpret_cast<size_t*>(shm_data_ptr + d.doc_sizes_offset), temp_doc_sizes.size());

    // copying doc_sizes_offset & doc_mem_offset into memory
    memcpy(shm_data_ptr + sizeof(size_t), &d.doc_sizes_offset, sizeof(size_t));
    memcpy(shm_data_ptr, &d.doc_mem_offset, sizeof(size_t));
}

void calc_rank_offsets(docs_container& d, char* shm_data_ptr, size_t world_size) {
    const size_t total_bytes = d.doc_mem.size();
    const size_t rank_count = min(world_size, d.doc_sizes.size());
    vector<size_t> temp_rank_offsets;
    temp_rank_offsets.reserve(rank_count);

    size_t bytes_remaining = total_bytes;
    size_t buckets_left = rank_count;
    size_t current_bucket_size = 0;
    size_t current_doc_offset = 0;
    size_t items_left = d.doc_sizes.size();

    for (size_t i = 0; i < d.doc_sizes.size(); ++i) {
        const size_t doc_size = d.doc_sizes[i];

        const double target = static_cast<double>(bytes_remaining) / static_cast<double>(buckets_left);
        const double size_if_added = static_cast<double>(current_bucket_size + doc_size);
        const double overshoot = size_if_added - target;
        const double undershoot = target - static_cast<double>(current_bucket_size);

        if (current_bucket_size > 0 &&
            buckets_left > 1 &&
            items_left >= buckets_left &&
            overshoot > undershoot) {
                // finalize the current bucket and move to the next one
                temp_rank_offsets.push_back(current_doc_offset - current_bucket_size);
                bytes_remaining -= current_bucket_size;
                --buckets_left;

                // new bucket takes overflow of last
                current_bucket_size = doc_size;
        } else {
            current_bucket_size += doc_size;
        }

        current_doc_offset += doc_size;
        --items_left;
    }

    // finalize the last bucket
    if (temp_rank_offsets.size() < rank_count) {
        temp_rank_offsets.push_back(total_bytes-current_bucket_size);
    }
    // assigning rank_offsets_offset
    d.rank_offsets_offset = d.doc_sizes_offset + d.doc_sizes.size() * sizeof(size_t);

    // copying in rank_offsets vector & assigning span
    memcpy(shm_data_ptr + d.rank_offsets_offset, temp_rank_offsets.data(), temp_rank_offsets.size() * sizeof(size_t));
    d.rank_offsets = span<size_t>(reinterpret_cast<size_t*>(shm_data_ptr + d.rank_offsets_offset), temp_rank_offsets.size());

    // assigning rank_offsets_offset into memory
    memcpy(shm_data_ptr + sizeof(size_t)*2, &d.rank_offsets_offset, sizeof(size_t));
}

void interpret_filled_docs_container(docs_container& d, char* shm_data_ptr, size_t world_size) {
    // interpreting the filled docs_container from shared memory
    d.doc_mem_offset = *((size_t*)shm_data_ptr);
    d.doc_sizes_offset = *((size_t*)shm_data_ptr + 1);
    d.rank_offsets_offset = *((size_t*)shm_data_ptr + 2);

    d.doc_mem = span<char>(shm_data_ptr + d.doc_mem_offset, d.doc_sizes_offset - d.doc_mem_offset);
    d.doc_sizes = span<size_t>(reinterpret_cast<size_t*>(shm_data_ptr + d.doc_sizes_offset), d.rank_offsets_offset - d.doc_sizes_offset);
    // big assumption made that there will always be more docs then ranks, thus no empty ranks
    d.rank_offsets = span<size_t>(reinterpret_cast<size_t*>(shm_data_ptr + d.rank_offsets_offset), world_size);
}

void fill_work_container(work_container& w, docs_container& d, char* shm_data_ptr, int rank) {
    // accounting for edge case of last rank
    if (rank == d.rank_offsets.size() - 1) {
        w.work_mem_size = (d.doc_sizes_offset - d.doc_mem_offset) - d.rank_offsets[rank];
    } else {
        w.work_mem_size = d.rank_offsets[rank + 1] - d.rank_offsets[rank];
    }

    // creating the work memory span based on the calculated size and offset
    w.work_mem = span<char>(shm_data_ptr + d.doc_mem_offset + d.rank_offsets[rank], w.work_mem_size);
}