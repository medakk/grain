#pragma once

#include "util.h"

namespace grain {
class GPUImage {
public:
    GPUImage(size_t N) : m_N(N) {
        cuda_assert(cudaMallocManaged(&m_image, N * N * sizeof(unsigned int)));
    }

    ~GPUImage() {
        cuda_assert(cudaFree(m_image));
    }

    ///////////////////
    // getters
    unsigned int *data() { return m_image; }
    size_t width() const { return m_N; }
    size_t height() const { return m_N; }

    ///////////////////
    // image operations
    void fill(unsigned int val);
    void fill(size_t row, size_t col, size_t n_rows, size_t n_cols, unsigned int val);

    ///////////////////
    // sync GPU ops
    void sync() {
        cuda_assert(cudaDeviceSynchronize());
    }

    // Disable copying and assignment
    GPUImage(const GPUImage &) = delete;
    GPUImage(const GPUImage &&) = delete;
    GPUImage &operator=(const GPUImage &) = delete;
    GPUImage &operator=(const GPUImage &&) = delete;

private:
    unsigned int *m_image;
    size_t m_N;
};
}