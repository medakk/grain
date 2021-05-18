#include <cassert>
#include "grain_sim.h"

namespace grain {
__global__ void gpu_step(const uint32_t* in, uint32_t* out, size_t n) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = x + y * n;
        uint32_t cell_out = GrainType::Undefined;

        if(in[idx] == GrainType::Blank) {
            auto d_blocked = y == n-1 || in[x + (y+1)*n] == GrainType::Sand;

            if(y != 0 && in[x + (y-1)*n] == GrainType::Sand) {
                cell_out = GrainType::Sand;
            } else if (d_blocked && x != n - 1
                       && in[x + 1 + (y - 1)*n] == GrainType::Sand
                       && in[x + 1 + y * n] == GrainType::Sand) {
                cell_out = GrainType::Sand;
            } else if (d_blocked && x != 0
                       && in[x - 1 + (y - 1) * n] == GrainType::Sand
                       && in[x - 1 + y * n] == GrainType::Sand) {
                cell_out = GrainType::Sand;
            } else {
                cell_out = GrainType::Blank;
            }
        } else if(in[idx] == GrainType::Sand) {
            auto dl_blocked = x == 0 || y == n - 1 || in[x-1 + (y+1)*n] == GrainType::Sand;
            auto dr_blocked = x == n-1 || y == n - 1 || in[x+1 + (y+1)*n] == GrainType::Sand;
            if(y != n-1 && in[x + (y+1)*n] == GrainType::Blank) {
                cell_out = GrainType::Blank;
            } else if(!dl_blocked) {
                cell_out = GrainType::Blank;
            } else if(!dr_blocked) {
                cell_out = GrainType::Blank;
            } else {
                cell_out = GrainType::Sand;
            }
        }

        out[idx] = cell_out;
    }
}

__global__ void gpu_sprinkle(uint32_t *out, size_t n, uint32_t value,
                             size_t start_x, size_t start_y, size_t sz) {
    auto x = start_x + blockIdx.x * blockDim.x + threadIdx.x;
    auto y = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n && x < start_x + sz && y < start_y + sz) {
        auto idx = x + y * n;
        // todo use curand
        if(x % 2 == 0 && y % 2 == 0) {
            out[idx] = value;
        }
    }
}

void GrainSim::step(const GPUImage& in, GPUImage& out) {
    const size_t T = 16;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((m_N + T - 1) / T, (m_N + T - 1) / T);

    gpu_step<<<numBlocks, threadsPerBlock>>>(in.data(), out.data(), m_N);

    cuda_assert(cudaPeekAtLastError());
}

void GrainSim::sprinkle(grain::GPUImage &image, uint32_t value,
                        size_t x, size_t y, size_t sz) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((sz + 16 - 1)/16, (sz + 16 - 1)/16);
    gpu_sprinkle<<<numBlocks, threadsPerBlock>>>(image.data(), image.width(), value,
                                                 x, y, sz);

    cuda_assert(cudaPeekAtLastError());

}
}
