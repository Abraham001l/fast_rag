#pragma once
#include <atomic>
#include <cstddef>
using namespace std;

template <typename T, size_t container_size>
class queue_container {
private:
    // atomic vars remain lock_free across separate MPI processes
    atomic<size_t> head;
    atomic<size_t> tail;
    T buffer[container_size];

public:
    bool is_empty() const;
    bool is_full() const;
    bool try_push(const T& item);
    bool try_pop(T& item);
}

template <typename T, size_t container_size>
bool queue_container<T, container_size>::is_empty() const {
    return head.load(memory_order_acquire) == tail.load(memory_order_acquire);
}

template <typename T, size_t container_size>
bool queue_container<T, container_size>::is_full() const {
    return (tail.load(memory_order_acquire) + 1) % container_size == head.load(memory_order_acquire);
}

template <typename T, size_t container_size>
bool queue_container<T, container_size>::try_push(const T& item) {
    size_t current_tail = tail.load(memory_order_relaxed);

    while (true) {
        // checking if the queue is full
        if ((current_tail + 1) % container_size == head.load(memory_order_acquire)) {
            return false;
        }

        // attempt to update the tail
        size_t next_tail = (current_tail + 1) % container_size;
        if (tail.compare_exchange_weak(current_tail, next_tail, memory_order_release, memory_order_relaxed)) {
            buffer[current_tail] = item;
            return true;
        }
    }
}

template <typename T, size_t container_size>
bool queue_container<T, container_size>::try_pop(T& item) {
    size_t current_head = head.load(memory_order_relaxed);

    while (true) {
        // checking if the queue is empty
        if (current_head == tail.load(memory_order_acquire)) {
            return false;
        }

        // attempt to update the head
        size_t next_head = (current_head + 1) % container_size;
        if (head.compare_exchange_weak(current_head, next_head, memory_order_release, memory_order_relaxed)) {
            item = buffer[current_head];
            return true;
        }
    }
}
