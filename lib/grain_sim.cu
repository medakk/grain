#include <cassert>

#include "gpu_grain_types.h"
#include "grain_types.h"
#include "grain_sim.h"
#include <fmt/format.h>

namespace grain {

__device__ void gpu_update_sandlike(grain_t* buf, size_t n, size_t turn,
                                    int x, int y, grain_t& val) {
    const auto type = val & GrainType::MASK_TYPE;
    if (y != n - 1) {
        if (is_passable(buf[x + (y + 1) * n])) {
            val = buf[x + (y + 1) * n] & GrainType::MASK_TYPE;
            buf[x + (y + 1) * n] = mark_done(type, turn);
        } else if (x != 0
                   && is_passable(buf[x - 1 + (y + 1) * n])) {
            val = buf[x - 1 + (y + 1) * n];
            buf[x - 1 + (y + 1) * n] = mark_done(type, turn);
        } else if (x != n - 1
                   && is_passable(buf[x + 1 + (y + 1) * n])) {
            val = buf[x + 1 + (y + 1) * n];
            buf[x + 1 + (y + 1) * n] = mark_done(type, turn);
        }
    }
}

__device__ void gpu_update_waterlike(grain_t *buf, size_t n, size_t turn,
                                     int x, int y, grain_t &val) {
    const auto type = val & GrainType::MASK_TYPE;
    if (y != n - 1 && is_type(buf[x + (y + 1) * n], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[x + (y + 1) * n] = mark_done(type, turn);
    } else if (y != n - 1 && x != 0
               && is_type(buf[x - 1 + (y + 1) * n], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[x - 1 + (y + 1) * n] = mark_done(type, turn);
    } else if (y != n - 1 && x != n - 1
               && is_type(buf[x + 1 + (y + 1) * n], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[x + 1 + (y + 1) * n] = mark_done(type, turn);
    } else if (x != 0
               && is_type(buf[x - 1 + y * n], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[x - 1 + y * n] = mark_done(type, turn);
    } else if (x != n - 1
               && is_type(buf[x + 1 + y * n], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[x + 1 + y * n] = mark_done(type, turn);
    }
}
__device__ void gpu_update_smokelike(grain_t* buf, size_t n, size_t turn,
                                     int x, int y, grain_t& val) {
    const auto type = val & GrainType::MASK_TYPE;
    if(y == 0) {
        val = GrainType::Blank;
    } else if(is_passable(buf[x + (y-1)*n])) {
        val = buf[x + (y - 1) * n] & GrainType::MASK_TYPE;

        // todo note that we are NOT marking it is as done. this is a hack to work
        // around this scenario:
        //
        // __S__
        // __S__
        //
        // ^these are two smoke particles. If, in our scheduling order, we update the
        // one on the bottom first, it'll move up, move the smoke above down, and mark
        // both as done. the end result is a no-op
        buf[x + (y - 1) * n] = type;
    }
}

__device__ void gpu_update_cell(grain_t* buf, size_t n, size_t turn, int x, int y) {
    auto idx = x + y * n;
    auto val = buf[idx];
    if (!is_done(val, turn)) {
        //todo switch-case?
        if (is_type(val, GrainType::Blank)) {

        } else if (is_type(val, GrainType::Sand)) {
            gpu_update_sandlike(buf, n, turn, x, y, val);
        } else if (is_type(val, GrainType::Water)) {
            gpu_update_waterlike(buf, n, turn, x, y, val);
        } else if (is_type(val, GrainType::Lava)) {
            bool done = false;
            // lava can destroy all the water/etc around it
            for(int dx=-1; dx<=1; dx++) {
                for(int dy=-1; dy<=1; dy++) {
                    const auto nx = x + dx;
                    const auto ny = y + dy;
                    if (nx >= 0 && nx < n - 1 && ny >= 0 && ny < n - 1
                        && is_type(buf[nx + ny * n], GrainType::Water)) {
                        val = GrainType::GrainType::Smoke;
                        buf[nx + ny * n] = mark_done(GrainType::Smoke, turn);
                        done = true;
                    }
                }
            }
            if(!done) {
                gpu_update_waterlike(buf, n, turn, x, y, val);
            }
        } else if (is_type(val, GrainType::Smoke)) {
            gpu_update_smokelike(buf, n, turn, x, y, val);
        }

        val = mark_done(val, turn);
        buf[idx] = val;
    }
}

__global__ void gpu_slow_step(grain_t *buf, size_t n, grain_t turn) {
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            gpu_update_cell(buf, n, turn, x, y);
        }
    }
}

__global__ void gpu_step(grain_t* buf, size_t n, grain_t turn, int bx, int by) {
    int sx = bx * 3 + blockIdx.x * blockDim.x * 6 + threadIdx.x * 6;
    int sy = by * 3 + blockIdx.y * blockDim.y * 6 + threadIdx.y * 6;
    for(int dx=0; dx<3; dx++) {
        for(int dy=0; dy<3; dy++) {
            auto x = sx + dx;
            auto y = sy + dy;
            if(x < n && y < n) {
#if 0
                grain_t col;
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

__global__ void gpu_sprinkle(grain_t *out, size_t n, grain_t value,
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

void GrainSim::sprinkle(grain::GPUImage &image, grain_t value,
                        size_t x, size_t y, size_t sz) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((sz + 16 - 1)/16, (sz + 16 - 1)/16);
    gpu_sprinkle<<<numBlocks, threadsPerBlock>>>(image.data(), image.width(), value,
                                                 x+sz/2, y+sz/2, sz);

    cuda_assert(cudaPeekAtLastError());

}
}
