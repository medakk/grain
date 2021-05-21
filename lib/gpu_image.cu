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

template<typename T, typename S>
__global__ void gpu_as_type(const T *in, S *out, size_t n) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = I(x, y);
        out[idx] = in[idx];
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
        auto idx = I(x, y);
        buf[idx] = value;
    }
}

template<typename T>
__global__ void gpu_count(T *buf, size_t n, T val, int* out) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = I(x, y);
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
void GPUImage<T>::fill(size_t start_x, size_t start_y, size_t w, size_t h, T val) {
    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((w + n_threads - 1) / n_threads, (h + n_threads - 1) / n_threads);

    gpu_fill_block<<<numBlocks, threadsPerBlock>>>(m_data, m_N,
                                                   start_x, start_y, w, h,
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
template<typename T>
template<typename S>
GPUImage<S> GPUImage<T>::as_type() const {
    GPUImage<S> new_image(m_N);

    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((m_N + n_threads - 1) / n_threads, (m_N + n_threads - 1) / n_threads);

    gpu_as_type<<<numBlocks, threadsPerBlock>>>(m_data, new_image.data(), m_N);

    cuda_assert(cudaPeekAtLastError());

    return new_image;
}

template GPUImage<uint32_t> GPUImage<uint8_t>::as_type() const;

}
