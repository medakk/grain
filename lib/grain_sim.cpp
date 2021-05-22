#include <utility>
#include "grain_sim.h"
#include "fmt/format.h"

namespace grain {

GrainSim::GrainSim(size_t N_, size_t speed_, std::string init_filename_)
        : m_N(N_), m_speed(speed_),
          m_init_filename(std::move(init_filename_)),
          m_image(N_) {
    init();

    // copy color map to GPU
    cuda_assert(cudaMallocManaged(&m_color_map, GrainType::MAX_TYPES * sizeof(uint32_t)));
    memcpy(m_color_map, GrainType::Colors,
           GrainType::MAX_TYPES * sizeof(uint32_t));
}

GrainSim::~GrainSim() {
    cuda_assert(cudaFree(m_color_map));
}

void GrainSim::init() {
    m_image.fill(grain::GrainType::Blank);

    if(m_init_filename.empty()) {
        // just add a block of sand for debugging
        m_image.fill(20, 20, 20, 20, GrainType::Sand);
    } else {
        m_image.read_png(m_init_filename);
    }

    // add a stone border so we don't have to think about array index overflow
    m_image.fill(0, 0, m_N, 1, GrainType::Stone);
    m_image.fill(0, 0, 1, m_N, GrainType::Stone);
    m_image.fill(0, m_N-1, m_N, 1, GrainType::Stone);
    m_image.fill(m_N-1, 0, 1, m_N, GrainType::Stone);

    // make sure everything is flushed
    m_image.sync();
}

const grain_t* GrainSim::update(EventData& event_data, bool verbose) {
    if(!event_data.paused) {
        m_frame_count++;
    }

    // check whether we should reset
    if(event_data.reset) {
        init();
        m_frame_count = 0;
        event_data.reset = false;
        return m_image.data();
    }

    if(!event_data.paused) {
        Timer timer;

        // perform update
        step();
        m_image.sync();

        if(verbose) {
            const double t = timer.elapsed();
            fmt::print("[F: {:7}] [iter_time:  {:6g}ms / {:6g}its/s] \n",
                       m_frame_count, t*1000.0, 1.0 / t);
        }
    }

    handle_brush_events(event_data);

    if(event_data.screenshot) {
        m_image.write_png("screenshot.png");
        event_data.screenshot = false;
    }

    return m_image.data();
}

void GrainSim::handle_brush_events(EventData& event_data) {
    //todo the event data being reset is kinda all over the place. some are here,
    // some are in the renderer


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
        sprinkle(m_image, brush, x, y, sz);
        m_image.sync();
    }
}

}