#pragma once

#include "grain_types.h"

//TODO this file exists because I grain_types.h seems to be compiled by gcc and not nvcc.
// this allows for these convenience methods to use __device__

namespace grain {
__device__ __host__ static bool is_type(uint32_t val, uint32_t type) {
    return (val & GrainType::MASK_TYPE) == type;
}

__device__ __host__ static bool is_done(uint32_t val, uint32_t turn) {
    return (val & GrainType::MASK_TURN) != turn;
}

[[nodiscard]]
__device__ __host__ static uint32_t mark_done(uint32_t val, uint32_t turn) {
    turn ^= 1; // flips the bit
    return (val & GrainType::MASK_TYPE) | turn;
}
}