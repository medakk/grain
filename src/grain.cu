#include "grain.h"
#include <cassert>

__global__ void gpu_step(const int* in, int* out, size_t n) {
}

__global__ void gpu_test_image(unsigned int *buf, size_t n) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    auto idx = x + y * n;
    buf[idx] = UINT_MAX;
}

// todo how to avoid the explicit need for these wrappers?

void Grain::step(const int* in, int *out, size_t n) {
}

void Grain::test_image(unsigned int *buf, size_t n) {
    assert(n%16 == 0);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n/16, n/16);
    gpu_test_image<<<numBlocks, threadsPerBlock>>>(buf, n);

    cuda_assert(cudaPeekAtLastError());
}
