#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

// shorthand for indexing
#define I(x, y) (x) + (y) * n
#define I_n(x, y, n) (x) + (y) * (n)

namespace grain {

template<class IteratorType>
void print_arr(IteratorType it, IteratorType end, char end_ch = '\n') {
    while (it != end) {
        std::cout << *it << " ";
        it++;
    }
    if (end_ch) {
        std::cout << end_ch;
    }
}

// $50 says one of my dependencies already has a version of this...
class Timer {
public:
    Timer() {
        m_start = std::chrono::system_clock::now();
    }

    double elapsed() {
        const auto end = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - m_start)
                .count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
};

#define cuda_assert(ans) { cuda_assert_((ans), __FILE__, __LINE__); }
inline void cuda_assert_(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

}