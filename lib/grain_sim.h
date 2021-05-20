#pragma once

#include <array>

#include "grain_types.h"
#include "gpu_image.h"

namespace grain{

class GrainSim {
public:
    using ImageType = GPUImage<grain_t>;

    GrainSim(size_t N_, size_t speed_ = 1, const std::string &init_filename_ = "")
            : m_N(N_), m_speed(speed_), m_init_filename(init_filename_) {
        // create two buffers for current state and previous state
        for(int i=0; i<2; i++) {
            m_images.emplace_back(m_N);
        }

        init();
    }

    const grain_t* update(EventData& event_data, bool verbose=true);

    // todo return the buffer as 4-channel image. should be removed after implemented in shader
    // return the buffer as 4-channel image
    GPUImage<uint32_t> as_color_image() const;

private:
    size_t m_N;
    size_t m_speed{1};
    std::vector<ImageType> m_images;
    size_t m_frame_count{0};
    size_t m_brush_idx{1}; // index for the current user-selected brush
    std::string m_init_filename{};

    // could be static, but that's a pain...
    std::array<grain_t, 5> m_brushes{
        GrainType::Blank,
        GrainType::Sand,
        GrainType::Water,
        GrainType::Lava,
        GrainType::Smoke,
    };

    // reset state
    void init();

    void step(const ImageType& in, ImageType& out);

    void handle_brush_events(ImageType& image, EventData& event_data);

    // these are generic operations. perhaps they can be moved to a different place later
    static void sprinkle(ImageType& image, grain_t value,
                         size_t x, size_t y,size_t sz);
};

}