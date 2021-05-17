#pragma once

#include "grain_types.h"
#include "gpu_image.h"

namespace grain{

class GrainSim {
public:
    GrainSim(size_t N_) : m_N(N_) {
        // create two buffers for current state and previous state
        for(int i=0; i<2; i++) {
            m_images.emplace_back(m_N);
        }

        init(m_images[0]);
    }

    const uint32_t* update(size_t frame_counter, EventData& event_data);

private:
    size_t m_N;
    std::vector<GPUImage> m_images;

    void step(const GPUImage& in, GPUImage& out);

    // these are generic operations. perhaps they can be moved to a different place later
    static void init(grain::GPUImage& image);
    static void sprinkle(grain::GPUImage& image, uint32_t value,
                         size_t x, size_t y,size_t sz);
};

}