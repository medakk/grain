#include <cassert>

#include "gpu_image.h"

namespace grain {
__global__ void gpu_fill(unsigned int *buf, size_t n,
                         unsigned int val) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    auto idx = x + y * n;
    buf[idx] = val;
}
__global__ void gpu_fill_block(unsigned int *buf, size_t n,
                               size_t row, size_t col,
                               size_t n_rows, size_t n_cols,
                               unsigned int val) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= row && x < row + n_rows && y >= col && y < col + n_cols) {
        auto idx = x + y * n;
        buf[idx] = val;
    }
}

void GPUImage::fill(unsigned int val) {
    assert(m_N%16 == 0);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(m_N/16, m_N/16);
    gpu_fill<<<numBlocks, threadsPerBlock>>>(m_image, m_N, val);

    cuda_assert(cudaPeekAtLastError());
}

void GPUImage::fill(size_t row, size_t col, size_t n_rows, size_t n_cols, unsigned int val) {
    assert(m_N%16 == 0);

    //todo yes this is very inefficient. its just to add colors to debug stuff. maybe should
    // just do it on CPU, its managed memory anyway. But where's the fun in that
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(m_N / 16, m_N / 16);
    gpu_fill_block<<<numBlocks, threadsPerBlock>>>(m_image, m_N,
                                                   row, col, n_rows, n_cols,
                                                   val);

    cuda_assert(cudaPeekAtLastError());
}

}
