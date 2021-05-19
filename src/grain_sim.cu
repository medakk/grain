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
                if (is_passable(buf[x + (y + 1) * n])) {
                    val = buf[x + (y + 1) * n] & GrainType::MASK_TYPE;
                    buf[x + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                } else if (x != 0
                           && is_passable(buf[x - 1 + (y + 1) * n])) {
                    val = buf[x - 1 + (y + 1) * n];
                    buf[x - 1 + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                } else if (x != n - 1
                           && is_passable(buf[x + 1 + (y + 1) * n])) {
                    val = buf[x + 1 + (y + 1) * n];
                    buf[x + 1 + (y + 1) * n] = mark_done(GrainType::Sand, turn);
                }
            }
        } else if (is_type(val, GrainType::Water)) {
            if (y != n - 1 && is_type(buf[x + (y + 1) * n], GrainType::Blank)) {
                val = GrainType::Blank;
                buf[x + (y + 1) * n] = mark_done(GrainType::Water, turn);
            } else if (y != n - 1 && x != 0
                       && is_type(buf[x - 1 + (y + 1) * n], GrainType::Blank)) {
                val = GrainType::Blank;
                buf[x - 1 + (y + 1) * n] = mark_done(GrainType::Water, turn);
            } else if (y != n - 1 && x != n - 1
                       && is_type(buf[x + 1 + (y + 1) * n], GrainType::Blank)) {
                val = GrainType::Blank;
                buf[x + 1 + (y + 1) * n] = mark_done(GrainType::Water, turn);
            } else if (x != 0
                       && is_type(buf[x - 1 + y * n], GrainType::Blank)) {
                val = GrainType::Blank;
                buf[x - 1 + y * n] = mark_done(GrainType::Water, turn);
            } else if (x != n - 1
                       && is_type(buf[x + 1 + y * n], GrainType::Blank)) {
                val = GrainType::Blank;
                buf[x + 1 + y * n] = mark_done(GrainType::Water, turn);
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

__global__ void gpu_step(uint32_t* buf, size_t n, uint32_t turn, size_t bx, size_t by) {
    auto sx = bx * 3 + blockIdx.x * blockDim.x * 6 + threadIdx.x * 6;
    auto sy = by * 3 + blockIdx.y * blockDim.y * 6 + threadIdx.y * 6;
    for(size_t dx=0; dx<3; dx++) {
        for(size_t dy=0; dy<3; dy++) {
            auto x = sx + dx;
            auto y = sy + dy;
            if(x < n && y < n) {
#if 0
                uint32_t col;
                if(bx == 0 && by == 0) {
                    col = GrainType::Debug0;
                } else if(bx == 0 && by == 1) {
                    col = GrainType::Debug1;
                } else if(bx == 1 && by == 0) {
                    col = GrainType::Debug2;
                } else {
                    col = GrainType::Debug3;
                }
                buf[x + y * n] = col;
#endif
                gpu_update_cell(buf, n, turn, x, y);
            }
        }
    }
}

__global__ void gpu_sprinkle(uint32_t *out, size_t n, uint32_t value,
                             size_t start_x, size_t start_y, size_t sz) {
    //todo this is very suspicious maybe some overflows but we are compensating later
    auto x = start_x - sz/2 + blockIdx.x * blockDim.x + threadIdx.x;
    auto y = start_y - sz/2 + blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n && x < start_x + sz && y < start_y + sz) {
        auto dx = start_x - x;
        auto dy = start_y - y;

        if(dx*dx + dy*dy <= sz * sz / 4) {
            auto idx = x + y * n;
            // todo use curand
            out[idx] = value;
        }

    }
}

void GrainSim::step(const GPUImage& in, GPUImage& out) {
    // todo find better way to do double buffer
    out = in; // GPU-copy from in to out

    const size_t T = 16;
    const size_t thirds = (m_N + 3 - 1) / 3;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((thirds + 2*T - 1) / (2 * T), (thirds + 2*T - 1) / (2 * T));

    // todo this is incorrect, we maybe wasting a full step on a noop
    auto turn = m_frame_count % 2;

    for(size_t i=0; i<m_speed; i++) {
        turn ^= 1;
        // gpu_slow_step<<<1, 1>>>(out.data(), m_N, turn);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 0, 0);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 0, 1);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 1, 0);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 1, 1);
    }

    cuda_assert(cudaPeekAtLastError());
}

void GrainSim::sprinkle(grain::GPUImage &image, uint32_t value,
                        size_t x, size_t y, size_t sz) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((sz + 16 - 1)/16, (sz + 16 - 1)/16);
    gpu_sprinkle<<<numBlocks, threadsPerBlock>>>(image.data(), image.width(), value,
                                                 x+sz/2, y+sz/2, sz);

    cuda_assert(cudaPeekAtLastError());

}
}
