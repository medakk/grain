#pragma once

#include "grain_types.h"

//TODO this file exists because grain_types.h seems to be compiled by gcc and not nvcc
// this allows for these convenience methods to use __device__. should ideally be one
// single header for grain_types(or not?)

namespace grain {
__device__ __host__ static bool is_type(grain_t val, grain_t type) {
    return (val & GrainType::MASK_TYPE) == type;
}

__device__ __host__ static bool is_passable(grain_t val) {
    val = val & GrainType::MASK_TYPE;
    return val == GrainType::Blank
           || val == GrainType::Water
           || val == GrainType::Lava
           || val == GrainType::Smoke;
}

__device__ __host__ static bool is_done(grain_t val, grain_t turn) {
    return (val & GrainType::MASK_TURN) != turn;
}

[[nodiscard]]
__device__ __host__ static grain_t mark_done(grain_t val, grain_t turn) {
    turn ^= GrainType::MASK_TURN;
    return (val & GrainType::MASK_TYPE) | turn;
}
}