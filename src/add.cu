#include "grain.h"

__global__ void gpu_add(const float *a_in, const float *b_in,
                        float *c_out, size_t n) {
    const auto i = threadIdx.x;
    c_out[i] = a_in[i] + b_in[i];
}

void Grain::add(const float *a_in, const float *b_in, float *c_out, size_t n) {
    gpu_add<<<1, n>>>(a_in, b_in, c_out, n);
}