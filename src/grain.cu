#include "grain.h"

__global__ void gpu_step(const int* in, int* out, size_t n) {
}

__global__ void gpu_test_image(int *buf, size_t n) {
    auto idx = threadIdx.x + threadIdx.y * n;
    buf[idx] = 1<<31;
}

// todo how to avoid the explicit need for these wrappers?

void Grain::step(const int* in, int *out, size_t n) {
}

void Grain::test_image(int *buf, size_t n) {
    dim3 threadsPerBlock(n, n);
    dim3 numBlocks;
    gpu_test_image<<<numBlocks, threadsPerBlock>>>(buf, n);
}
