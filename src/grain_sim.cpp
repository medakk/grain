#include "grain_sim.h"

namespace grain {

void GrainSim::init(grain::GPUImage& image) {
    image.fill(grain::GrainType::Blank);
    image.fill(20, 20, 20, 20, grain::GrainType::Sand);
    image.sync();
}

const uint32_t* GrainSim::update(size_t frame_counter, EventData& event_data) {
    // figure out which buffer is in vs out
    const auto &image0 = m_images[frame_counter % 2];
    auto& image1 = m_images[(frame_counter+1) % 2];

    // check whether we should reset
    if(event_data.should_reset) {
        init(image1);
        event_data.should_reset = false;
        return image1.data();
    }

    // perform update
    step(image0, image1);

    // handle mouse events
    if(event_data.mouse_pressed) {
        // std::cout << "mouse: " << event_data.mouse_x << " " << event_data.mouse_y << "\n";
        const size_t x = event_data.mouse_x * (m_N - 1);
        const size_t y = event_data.mouse_y * (m_N - 1);
        const size_t sz = 15;
        sprinkle(image1, GrainType::Sand, x, y, sz);
    }

    image1.sync();
    return image1.data();
}

}