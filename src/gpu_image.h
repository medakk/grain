#pragma once

#include "util.h"

namespace grain {
class GPUImage {
public:
    GPUImage(size_t N) : m_N(N) {
        cuda_assert(cudaMallocManaged(&m_image, N * N * sizeof(uint32_t)));
    }

    ~GPUImage() {
        cuda_assert(cudaFree(m_image));
    }

    ///////////////////
    // getters
    uint32_t *data() { return m_image; }
    size_t width() const { return m_N; }
    size_t height() const { return m_N; }

    ///////////////////
    // image operations
    void fill(uint32_t val);
    void fill(size_t row, size_t col, size_t n_rows, size_t n_cols, uint32_t val);

    ///////////////////
    // sync GPU ops
    void sync() {
        cuda_assert(cudaDeviceSynchronize());
    }

    //////////////////////////////////
    // Disable copying and assignment
    GPUImage(const GPUImage &) = delete;
    GPUImage(const GPUImage &&) = delete;
    GPUImage &operator=(const GPUImage &) = delete;
    GPUImage &operator=(const GPUImage &&) = delete;

private:
    uint32_t *m_image;
    size_t m_N;
};
}