#pragma once

#include "grain_types.h"
#include "gpu_image.h"

namespace grain{

class GrainSim {
public:
    static void step(const GPUImage& in, GPUImage& out, size_t N);
};
}