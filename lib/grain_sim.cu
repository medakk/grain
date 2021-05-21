#include <cassert>

#include "gpu_grain_types.h"
#include "grain_types.h"
#include "grain_sim.h"
#include <fmt/format.h>

namespace grain {

// each thread operates on a THREAD_DIM x THREAD_DIM square
const size_t THREAD_DIM = 2;

__device__ void gpu_update_sandlike(grain_t* buf, size_t n, size_t turn,
                                    int x, int y, grain_t& val) {
    const auto type = val & GrainType::MASK_TYPE;
    if (y != n - 1) {
        if (is_passable(buf[I(x, y+1)])) {
            val = buf[I(x, y+1)] & GrainType::MASK_TYPE;
            buf[I(x, y+1)] = mark_done(type, turn);
        } else if (x != 0
                   && is_passable(buf[I(x-1, y+1)])) {
            val = buf[I(x-1, y+1)];
            buf[I(x-1, y+1)] = mark_done(type, turn);
        } else if (x != n - 1
                   && is_passable(buf[I(x+1, y+1)])) {
            val = buf[I(x+1, y+1)];
            buf[I(x+1, y+1)] = mark_done(type, turn);
        }
    }
}

__device__ void gpu_update_waterlike(grain_t *buf, size_t n, size_t turn,
                                     int x, int y, grain_t &val) {
    const auto type = val & GrainType::MASK_TYPE;
    if (y != n - 1 && is_type(buf[I(x, y+1)], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[I(x, y+1)] = mark_done(type, turn);
    } else if (y != n - 1 && x != 0
               && is_type(buf[I(x-1, y+1)], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[I(x-1, y+1)] = mark_done(type, turn);
    } else if (y != n - 1 && x != n - 1
               && is_type(buf[I(x+1, y+1)], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[I(x+1, y+1)] = mark_done(type, turn);
    } else if (x != 0
               && is_type(buf[I(x-1, y)], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[I(x-1, y)] = mark_done(type, turn);
    } else if (x != n - 1
               && is_type(buf[I(x+1, y)], GrainType::Blank)) {
        val = GrainType::Blank;
        buf[I(x+1, y)] = mark_done(type, turn);
    }
}
__device__ void gpu_update_smokelike(grain_t* buf, size_t n, size_t turn,
                                     int x, int y, grain_t& val) {
    const auto type = val & GrainType::MASK_TYPE;
    if(y == 0) {
        val = GrainType::Blank;
    } else if(is_passable(buf[I(x, y-1)])) {
        val = buf[I(x, y-1)] & GrainType::MASK_TYPE;

        // todo note that we are NOT marking it is as done. this is a hack to work
        // around this scenario:
        //
        // __S__
        // __S__
        //
        // ^these are two smoke particles. If, in our scheduling order, we update the
        // one on the bottom first, it'll move up, move the smoke above down, and mark
        // both as done. the end result is a no-op
        buf[I(x, y-1)] = type;
    }
}
__device__ void gpu_update_lava(grain_t* buf, size_t n, size_t turn,
                                     int x, int y, grain_t& val) {
    bool done = false;

    // lava can destroy all the water/etc around it.
    for(int dx=-1; dx<=1; dx++) {
        for(int dy=-1; dy<=1; dy++) {
            const auto nx = x + dx;
            const auto ny = y + dy;
            if (nx >= 0 && nx < n - 1 && ny >= 0 && ny < n - 1
                && is_type(buf[I(nx, ny)], GrainType::Water)) {
                val = GrainType::GrainType::Smoke;
                buf[I(nx, ny)] = mark_done(GrainType::Smoke, turn);
                done = true;
            }
        }
    }

    // if there is nothing to destroy, behave like water.
    if(!done) {
        gpu_update_waterlike(buf, n, turn, x, y, val);
    }
}

__device__ void gpu_update_cell(grain_t* buf, size_t n, size_t turn, int x, int y) {
    auto idx = I(x, y);
    auto val = buf[idx];
    if (!is_done(val, turn)) {
        //todo switch-case?
        if (is_type(val, GrainType::Blank)) {

        } else if (is_type(val, GrainType::Sand)) {
            gpu_update_sandlike(buf, n, turn, x, y, val);
        } else if (is_type(val, GrainType::Water)) {
            gpu_update_waterlike(buf, n, turn, x, y, val);
        } else if (is_type(val, GrainType::Lava)) {
            gpu_update_lava(buf, n, turn, x, y, val);
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
    int sx = bx * THREAD_DIM + blockIdx.x * blockDim.x * THREAD_DIM * 2 + threadIdx.x * THREAD_DIM * 2;
    int sy = by * THREAD_DIM + blockIdx.y * blockDim.y * THREAD_DIM * 2 + threadIdx.y * THREAD_DIM * 2;
    for(int dx=0; dx<THREAD_DIM; dx++) {
        for(int dy=0; dy<THREAD_DIM; dy++) {
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
#else
                gpu_update_cell(buf, n, turn, x, y);
#endif
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
            auto idx = I(x, y);
            // todo use curand
            out[idx] = value;
        }

    }
}

__global__ void gpu_as_color_image(const grain_t *in, uint32_t *out,
                                   size_t n, const uint32_t *map) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = I(x, y);
        out[idx] = map[in[idx] & GrainType::MASK_TYPE];
    }
}

void GrainSim::step(const GrainSim::ImageType& in, GrainSim::ImageType& out) {
    // todo find better way to do double buffer
    out = in; // GPU-copy from in to out

    const size_t T = 16;
    const size_t divisions = (m_N + THREAD_DIM - 1) / THREAD_DIM;
    dim3 threadsPerBlock(T, T);
    dim3 numBlocks((divisions + 2*T - 1) / (2 * T), (divisions + 2*T - 1) / (2 * T));

    // todo this is incorrect, we maybe wasting a full step on a noop. also this is a hack
    static_assert(GrainType::MASK_TURN == 0x80, "you changed MASK_TURN but forgot to fix this hack");
    auto turn = (m_frame_count % 2) << 7;

    for(size_t i=0; i<m_speed; i++) {
        turn ^= GrainType::MASK_TURN;
        // gpu_slow_step<<<1, 1>>>(out.data(), m_N, turn);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 0, 0);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 0, 1);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 1, 0);
        gpu_step<<<numBlocks, threadsPerBlock>>>(out.data(), m_N, turn, 1, 1);
    }

    cuda_assert(cudaPeekAtLastError());
}

void GrainSim::sprinkle(GrainSim::ImageType &image, grain_t value,
                        size_t x, size_t y, size_t sz) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((sz + 16 - 1)/16, (sz + 16 - 1)/16);
    gpu_sprinkle<<<numBlocks, threadsPerBlock>>>(image.data(), image.width(), value,
                                                 x+sz/2, y+sz/2, sz);

    cuda_assert(cudaPeekAtLastError());
}


void GrainSim::as_color_image(GPUImage<uint32_t>& image_out) const {
    const auto& image_in = m_images[(m_frame_count+1) % 2];
    assert(image_in.width() == image_out.width() && image_in.height() == image_out.height());

    const size_t n_threads = 16;
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 numBlocks((m_N + n_threads - 1) / n_threads, (m_N + n_threads - 1) / n_threads);

    static_assert(GrainType::MASK_TYPE == 0x1f, "you changed mask type but didn't verify if the color stuff still works");
    gpu_as_color_image<<<numBlocks, threadsPerBlock>>>(image_in.data(), image_out.data(),
                                                       m_N, m_color_map);
}

}
