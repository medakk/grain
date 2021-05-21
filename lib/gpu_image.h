#pragma once

#include "util.h"


namespace grain {

template <typename T>
class GPUImage {
public:
    GPUImage(size_t N) : m_N(N) {
        cuda_assert(cudaMallocManaged(&m_data, N * N * sizeof(T)));
    }

    ~GPUImage() {
        free_image();
    }

    ///////////////////
    // getters
    T *data() { return m_data; }
    const T *data() const { return m_data; }
    size_t width() const { return m_N; }
    size_t height() const { return m_N; }

    ///////////////////
    // image operations
    void fill(T val);
    void fill(size_t row, size_t col, size_t n_rows, size_t n_cols, T val);
    size_t count(T val) const;

    void write_png(const std::string& filename) const;
    void read_png(const std::string& filename);

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

        if(m_data == other.m_data) {
            //todo what if they are different views into the same buffer :p
            return *this;
        }

        if(m_N != other.m_N) {
            free_image();
            m_N = other.m_N;
            cuda_assert(cudaMallocManaged(&m_data, m_N * m_N * sizeof(T)));
        }

        cuda_assert(cudaMemcpy(m_data, other.m_data,
                               m_N * m_N * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
        sync();

        return *this;
    }

    // Move c'tor/assignment
    GPUImage(GPUImage &&other) {
        m_data = other.m_data;
        m_N = other.m_N;

        // make sure `other` has no more references to this GPU buffer
        other.m_data = nullptr;
    }

    GPUImage &operator=(GPUImage &&other) noexcept {
        if(this == &other) {
            return *this;
        }

        if(m_data == other.m_data) {
            //todo what if they are different views into the same buffer :p
            return *this;
        }

        free_image();
        m_data = other.m_data;
        m_N = other.m_N;

        other.m_data = nullptr;
        return *this;
    }

    template <typename S>
    GPUImage<S> as_type() const;

private:
    T *m_data;
    size_t m_N;

    void free_image() {
        if(m_data != nullptr) {
            cuda_assert(cudaFree(m_data));
        }
    }

    int n_components() const;
};

// Explicit initialization for scalar types
template class GPUImage<uint8_t>;
template class GPUImage<uint32_t>;

}