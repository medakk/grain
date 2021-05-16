#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "grain.h"
#include "util.h"

int main() {
    const size_t N = 20;
    float *a{nullptr};
    float *b{nullptr};
    float *c{nullptr};

    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));

    assert(a != nullptr);
    assert(b != nullptr);
    assert(c != nullptr);

    std::iota(a, &a[N], 1.0f);
    std::iota(b, &b[N], 10.0f);

    Grain::add(a, b, c, N);

    std::cout << "a: ";
    print_arr(a, &a[N]);
    std::cout << "b: ";
    print_arr(b, &b[N]);
    std::cout << "c: ";
    print_arr(c, &c[N]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
