#include <cuda_runtime.h>

#include "renderer.h"
#include "grain_types.h"
#include "grain_sim.h"

int main() {
    // world size
    const size_t N = 256;

    grain::GrainSim grain_sim(N);

    // create renderer and start update loop
    grain::MiniFBRenderer::start([&](size_t frame_counter,
                                     grain::EventData &event_data) {
        const uint32_t* data = grain_sim.update(frame_counter, event_data);
        return data;
    }, N, N);

    return 0;
}
