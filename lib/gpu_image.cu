#include <cassert>

#include "gpu_image.h"

namespace grain {

template<typename T>
__global__ void gpu_fill(T *buf, size_t n,
                         T val) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = x + y * n;
        buf[idx] = val;
    }
}

template<typename T>
__global__ void gpu_fill_block(T *buf, size_t n,
                               size_t start_x, size_t start_y,
                               size_t w, size_t h,
                               T value) {
    auto x = start_x + blockIdx.x * blockDim.x + threadIdx.x;
    auto y = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n && x < start_x + w && y < start_y + h) {
        auto idx = x + y * n;
        buf[idx] = value;
    }
}

template<typename T>
__global__ void gpu_count(T *buf, size_t n, T val, int* out) {
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

template<typename T>
void GPUImage<T>::fill(T val) {
    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((m_N + n_threads - 1) / n_threads, (m_N + n_threads - 1) / n_threads);
    gpu_fill<<<numBlocks, threadsPerBlock>>>(m_data, m_N, val);

    cuda_assert(cudaPeekAtLastError());
}

template<typename T>
void GPUImage<T>::fill(size_t row, size_t col, size_t n_rows, size_t n_cols, T val) {
    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((n_rows + n_threads - 1) / n_threads, (n_cols + n_threads - 1) / n_threads);

    gpu_fill_block<<<numBlocks, threadsPerBlock>>>(m_data, m_N,
                                                   row, col, n_rows, n_cols,
                                                   val);

    cuda_assert(cudaPeekAtLastError());
}

template<typename T>
size_t GPUImage<T>::count(T val) const {
    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((m_N + n_threads - 1) / n_threads, (m_N + n_threads - 1) / n_threads);

    int* d_out;
    cuda_assert(cudaMallocManaged(&d_out, sizeof(int)));
    *d_out = 0;

    gpu_count<<<numBlocks, threadsPerBlock>>>(m_data, m_N, val, d_out);

    cuda_assert(cudaPeekAtLastError());
    cuda_assert(cudaDeviceSynchronize());

    size_t out = *d_out;
    cuda_assert(cudaFree(d_out));

    return out;
}

}
