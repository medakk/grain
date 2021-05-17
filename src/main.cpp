#include <cuda_runtime.h>

#include "gpu_image.h"
#include "renderer.h"
#include "grain_types.h"
#include "grain_sim.h"

void init_sim(grain::GPUImage& image) {
    image.fill(grain::GrainType::Blank);
    image.fill(20, 20, 20, 20, grain::GrainType::Sand);
    image.sync();
}

int main() {
    // world size
    const size_t N = 256;

    // create two buffers for current state and previous state
    std::vector<grain::GPUImage> images;
    for(int i=0; i<2; i++) {
        images.emplace_back(N);
    }

    // initialize first image. second doesn't matter will be overwritten anyway
    init_sim(images[0]);

    // create renderer and start update loop
    grain::MiniFBRenderer::start([&](size_t frame_counter, bool shouldReset) {
        const auto& image0 = images[frame_counter % 2];
        auto& image1 = images[(frame_counter+1) % 2];

        if(shouldReset) {
            init_sim(image1);
        } else {
            grain::GrainSim::step(image0, image1, N);
            image1.sync();
        }

        return image1.data();
    }, N, N);

    return 0;
}
