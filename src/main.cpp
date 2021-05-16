#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_image.h"
#include "renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    const size_t N = 256;
    grain::GPUImage image(N);

    int x = 0;
    grain::MiniFBRenderer renderer(N, N);
    renderer.start([&]() {
        x = rand() % (N - 50);
        image.fill((0xff<<24) + (0xff<<8));
        image.fill(x, 10, 16, 16, (0xff<<24) + (0xff<<16));
        image.sync();

        return image.data();
    });

    assert(stbi_write_png("out.png", N, N, 4, image.data(), sizeof(uint32_t)*N) != 0);

    return 0;
}
