#pragma once

#include "util.h"

namespace grain {
class GPUImage {
public:
    GPUImage(size_t N) : m_N(N) {
        cuda_assert(cudaMallocManaged(&m_image, N * N * sizeof(uint32_t)));
    }

    ~GPUImage() {
        free_image();
    }

    ///////////////////
    // getters
    uint32_t *data() { return m_image; }
    const uint32_t *data() const { return m_image; }
    size_t width() const { return m_N; }
    size_t height() const { return m_N; }

    ///////////////////
    // image operations
    void fill(uint32_t val);
    void fill(size_t row, size_t col, size_t n_rows, size_t n_cols, uint32_t val);
    size_t count(uint32_t val) const;

    void write_png(const std::string& filename) const;

    ///////////////////
    // sync GPU ops (todo should this be const?)
    void sync() const {
        cuda_assert(cudaDeviceSynchronize());
    }

    //////////////////////////////////
    // Copy c'tor/assignment
    GPUImage(const GPUImage &) = delete;
    GPUImage &operator=(const GPUImage &other) {
        if(this == &other) {
            return *this;
        }

        if(m_image == other.m_image) {
            return *this;
        }

        if(m_N != other.m_N) {
            free_image();
            m_N = other.m_N;
            cuda_assert(cudaMallocManaged(&m_image, m_N * m_N * sizeof(uint32_t)));
        }

        cuda_assert(cudaMemcpy(m_image, other.m_image,
                               m_N * m_N * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyDefault));
        sync();

        return *this;
    }

    // Move c'tor/assignment
    GPUImage(GPUImage &&other) {
        m_image = other.m_image;
        m_N = other.m_N;

        // make sure `other` has no more references to this GPU buffer
        other.m_image = nullptr;
    }

    GPUImage &operator=(GPUImage &&other) noexcept {
        free_image();
        m_image = other.m_image;
        m_N = other.m_N;

        other.m_image = nullptr;
        return *this;
    }

private:
    uint32_t *m_image;
    size_t m_N;

    void free_image() {
        if(m_image != nullptr) {
            cuda_assert(cudaFree(m_image));
        }
    }
};
}