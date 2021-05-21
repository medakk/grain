#pragma once

#include <array>

#include "grain_types.h"
#include "gpu_image.h"

namespace grain{

class GrainSim {
public:
    using ImageType = GPUImage<grain_t>;

    explicit GrainSim(size_t N_, size_t speed_ = 1, std::string init_filename_ = "");
    ~GrainSim();

    const grain_t* update(EventData& event_data, bool verbose=true);

    // todo return the buffer as 4-channel image. should be removed after implemented in shader
    // return the buffer as 4-channel image
    void as_color_image(GPUImage<uint32_t>&) const;

    //////////////////////////////////
    // Disable copying and assignment
    GrainSim(const GrainSim &) = delete;
    GrainSim &operator=(const GrainSim &) = delete;
    GrainSim(GrainSim &&) = delete;
    GrainSim &operator=(GrainSim &&) = delete;

private:
    size_t m_N;
    size_t m_speed{1};
    ImageType m_image;
    size_t m_frame_count{0};
    size_t m_brush_idx{1}; // index for the current user-selected brush
    std::string m_init_filename{};

    uint32_t *m_color_map{nullptr};

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

    void step();

    void handle_brush_events(EventData& event_data);

    // these are generic operations. perhaps they can be moved to a different place later
    static void sprinkle(ImageType& image, grain_t value,
                         size_t x, size_t y,size_t sz);
};

}