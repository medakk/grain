#pragma once

class GPUImage {
public:
    GPUImage(size_t N) : m_N(N) {
        cuda_assert(cudaMallocManaged(&m_image, N * N * sizeof(unsigned int)));
    }

    ~GPUImage() {
        cuda_assert(cudaFree(m_image));
    }

    unsigned int* data() {
        return m_image;
    }

    // Disable copying and assignment
    GPUImage(const GPUImage&) = delete;
    GPUImage(const GPUImage&&) = delete;
    GPUImage& operator=(const GPUImage&) = delete;
    GPUImage& operator=(const GPUImage&&) = delete;

private:
    unsigned int *m_image;
    size_t m_N;
};
