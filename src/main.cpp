#include <cuda_runtime.h>

#include "gpu_image.h"
#include "renderer.h"
#include "grain.h"

int main() {
    const size_t N = 256;

    std::vector<grain::GPUImage> frames;
    frames.emplace_back(N);
    frames.emplace_back(N);

    frames[0].fill(grain::GrainType::BLANK);
    frames[1].fill(grain::GrainType::SAND);
    std::for_each(frames.begin(), frames.end(), [](auto& f){ f.sync(); });

    grain::MiniFBRenderer renderer(N, N);
    renderer.start([&](size_t frame_counter) {
        return frames[frame_counter % 2].data();
    });

    return 0;
}
