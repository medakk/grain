#include <chrono>
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

const uint32_t* GrainSim::update(EventData& event_data, bool verbose) {
    if(!event_data.paused) {
        m_frame_count++;
    }

    // figure out which buffer is in vs out
    const auto &image0 = m_images[m_frame_count % 2];
    auto& image1 = m_images[(m_frame_count+1) % 2];

    // check whether we should reset
    if(event_data.reset) {
        init();
        m_frame_count = 0;
        event_data.reset = false;
        return image1.data();
    }

    if(!event_data.paused) {
        using namespace std::chrono;
        const auto start_time = system_clock::now();

        // perform update
        step(image0, image1);

        const auto end_time = system_clock::now();
        const double elapsed_seconds = duration_cast<duration<double>>(
                end_time - start_time).count();
        if(verbose) {
            fmt::print("[F: {:7}] [iter_time: {:.6}ms] \n", m_frame_count, elapsed_seconds*1000.0);
        }
    }

    handle_brush_events(image1, event_data);

    image1.sync();

    if(event_data.screenshot) {
        image1.write_png("screenshot.png");
        event_data.screenshot = false;
    }

    return image1.data();
}

void GrainSim::handle_brush_events(GPUImage& image, EventData& event_data) {
    // brush change events
    if(event_data.next_brush) {
        m_brush_idx = (m_brush_idx + 1) % m_brushes.size();
        event_data.next_brush = false;
    }
    if(event_data.prev_brush) {
        m_brush_idx = (m_brush_idx + m_brushes.size() - 1) % m_brushes.size();
        event_data.prev_brush = false;
    }

    // handle mouse events
    if(event_data.mouse_pressed) {
        const size_t sz = 30.0 * (m_N / 512.0);
        const size_t x = event_data.mouse_x * (m_N - 1) - sz / 2.0;
        const size_t y = event_data.mouse_y * (m_N - 1) - sz / 2.0;
        const auto brush = m_brushes[m_brush_idx];
        sprinkle(image, brush, x, y, sz);
    }
}

}