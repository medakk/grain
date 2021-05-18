#include <cassert>

#include "gpu_image.h"

namespace grain {
__global__ void gpu_fill(uint32_t *buf, size_t n,
                         uint32_t val) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = x + y * n;
        buf[idx] = val;
    }
}
__global__ void gpu_fill_block(uint32_t *buf, size_t n,
                               size_t start_x, size_t start_y,
                               size_t w, size_t h,
                               uint32_t value) {
    auto x = start_x + blockIdx.x * blockDim.x + threadIdx.x;
    auto y = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n && x < start_x + w && y < start_y + h) {
        auto idx = x + y * n;
        buf[idx] = value;
    }
}
__global__ void gpu_count(uint32_t *buf, size_t n, uint32_t val, int* out) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = x + y * n;
        if(buf[idx] == val) {
            // todo parallel reduction?
            atomicAdd(out, 1);
        }
    }
}

void GPUImage::fill(uint32_t val) {
    const size_t T = 16;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((m_N + T - 1) / T, (m_N + T - 1) / T);
    gpu_fill<<<numBlocks, threadsPerBlock>>>(m_image, m_N, val);

    cuda_assert(cudaPeekAtLastError());
}

void GPUImage::fill(size_t row, size_t col, size_t n_rows, size_t n_cols, uint32_t val) {
    const size_t T = 16;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((n_rows + T - 1) / T, (n_cols + T - 1) / T);

    gpu_fill_block<<<numBlocks, threadsPerBlock>>>(m_image, m_N,
                                                   row, col, n_rows, n_cols,
                                                   val);

    cuda_assert(cudaPeekAtLastError());
}

size_t GPUImage::count(uint32_t val) const {
    const size_t T = 16;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((m_N + T - 1) / T, (m_N + T - 1) / T);

    int* d_out;
    cuda_assert(cudaMallocManaged(&d_out, sizeof(int)));
    *d_out = 0;

    gpu_count<<<numBlocks, threadsPerBlock>>>(m_image, m_N, val, d_out);

    cuda_assert(cudaPeekAtLastError());
    cuda_assert(cudaDeviceSynchronize());

    size_t out = *d_out;
    cuda_assert(cudaFree(d_out));

    return out;
}

}
