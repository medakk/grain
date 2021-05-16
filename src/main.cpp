#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "grain.h"
#include "util.h"
#include "gpu_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    const size_t N = 64;
    GPUImage image(N);

    Grain::test_image(image.data(), N);
    cuda_assert(cudaDeviceSynchronize());

    assert(stbi_write_png("out.png", N, N, 4, image.data(), sizeof(unsigned int)*N) != 0);

    return 0;
}
