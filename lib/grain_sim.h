#pragma once

#include <array>

#include "grain_types.h"
#include "gpu_image.h"

namespace grain{

class GrainSim {
public:
    GrainSim(size_t N_, size_t speed_=1) : m_N(N_), m_speed(speed_) {
        // create two buffers for current state and previous state
        for(int i=0; i<2; i++) {
            m_images.emplace_back(m_N);
        }

        init();
    }

    const uint32_t* update(EventData& event_data, bool verbose=true);


private:
    size_t m_N;
    size_t m_speed{1};
    std::vector<GPUImage> m_images;
    size_t m_frame_count{0};
    size_t m_brush_idx{0}; // index for the current user-selected brush

    // could be static, but that's a pain...
    std::array<uint32_t, 3> m_brushes{
        GrainType::Sand,
        GrainType::Blank,
        GrainType::Water,
    };

    // reset state
    void init();

    void step(const GPUImage& in, GPUImage& out);

    // these are generic operations. perhaps they can be moved to a different place later
    static void sprinkle(grain::GPUImage& image, uint32_t value,
                         size_t x, size_t y,size_t sz);
};

}