#include "grain_sim.h"
#include "fmt/format.h"

namespace grain {

void GrainSim::init() {
    m_images[0].fill(grain::GrainType::Blank);
    m_images[0].fill(20, 20, 20, 20, grain::GrainType::Sand);

    m_images[1] = m_images[0];

    m_images[0].sync();
    m_images[1].sync();
}

const uint32_t* GrainSim::update(EventData& event_data) {
    // figure out which buffer is in vs out
    const auto &image0 = m_images[m_frame_count % 2];
    auto& image1 = m_images[(m_frame_count+1) % 2];

    // check whether we should reset
    if(event_data.should_reset) {
        init();
        m_frame_count = 0;
        event_data.should_reset = false;
        return image1.data();
    }

    if(!event_data.is_paused) {
        const auto n_sand = image0.count(GrainType::Sand | (m_frame_count % 2));
        fmt::print("[F: {:7}] [sand: {:3}] \n", m_frame_count, n_sand);

        // perform update
        step(image0, image1);

        // handle mouse events
        if(event_data.is_mouse_pressed) {
            // std::cout << "mouse: " << event_data.mouse_x << " " << event_data.mouse_y << "\n";
            const size_t x = event_data.mouse_x * (m_N - 1);
            const size_t y = event_data.mouse_y * (m_N - 1);
            const size_t sz = 15;
            sprinkle(image1, GrainType::Sand, x, y, sz);
        }

        m_frame_count++;
    }

    image1.sync();
    return image1.data();
}

}