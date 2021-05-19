#include <cassert>

#include "gpu_grain_types.h"
#include "grain_types.h"
#include "grain_sim.h"
#include <fmt/format.h>

namespace grain {

__device__ void gpu_update_cell(uint32_t* buf, size_t n, size_t turn, size_t x, size_t y) {
    auto idx = x + y * n;
    auto val = buf[idx];
    if (!is_done(val, turn)) {
        //todo switch-case?
        if (is_type(val, GrainType::Blank)) {

        } else if (is_type(val, GrainType::Sand)) {
            if (y != n - 1) {
                if (is_type(buf[x + (y + 1) * n], GrainType::Blank)) {
                    val = GrainType::Blank;
                    buf[x + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                } else if (x != 0
                           && is_type(buf[x - 1 + (y + 1) * n], GrainType::Blank)) {
                    val = GrainType::Blank;
                    buf[x - 1 + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                } else if (x != n - 1
                           && is_type(buf[x + 1 + (y + 1) * n], GrainType::Blank)) {
                    val = GrainType::Blank;
                    buf[x + 1 + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                }
            }
        }

        val = mark_done(val, turn);
        buf[idx] = val;
    }
}

__global__ void gpu_slow_step(uint32_t *buf, size_t n, uint32_t turn) {
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            gpu_update_cell(buf, n, turn, x, y);
        }
    }
}

__global__ void gpu_step(uint32_t* buf, size_t n, uint32_t turn) {
    auto x = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
    auto y = blockIdx.y * blockDim.y * 3 + threadIdx.y * 3;
    if(x < n && y < n) {
        // gpu_update_cell(buf, n, turn, x, y);
        buf[x + y * n] = GrainType::Undefined;
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
    out = in;

    const size_t T = 16;
    const size_t thirds = (m_N + 3 - 1) / 3;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((thirds + T - 1) / T, (thirds + T - 1) / T);

    // gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, m_frame_count%2);
    gpu_slow_step<<<1, 1>>>(out.data(), m_N, m_frame_count%2);


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
