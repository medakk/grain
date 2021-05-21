#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

// shorthand for indexing
#define I(x, y) (x) + (y) * n

template<class IteratorType>
void print_arr(IteratorType it, IteratorType end, char end_ch='\n') {
    while(it != end) {
        std::cout << *it << " ";
        it++;
    }
    if(end_ch) {
        std::cout << end_ch;
    }
}

#define cuda_assert(ans) { cuda_assert_((ans), __FILE__, __LINE__); }
inline void cuda_assert_(cudaError_t code, const char *file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr,"cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
