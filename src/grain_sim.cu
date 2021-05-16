#include <cassert>
#include "grain_sim.h"

namespace grain {
__global__ void gpu_step(const uint32_t* in, uint32_t* out, size_t n) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < n) {
        auto idx = x + y * n;
        out[idx] = in[idx];
    }
}

void GrainSim::step(const GPUImage& in, GPUImage& out, size_t N) {
    assert(N%16 == 0);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N/16, N/16);
    gpu_step<<<numBlocks, threadsPerBlock>>>(in.data(), out.data(), N);

    cuda_assert(cudaPeekAtLastError());
}
}
