#include "gpu_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace grain {
void GPUImage::write_png(const std::string& filename) const {
    if(stbi_write_png(filename.c_str(), m_N, m_N, 4,
                      m_data, 0) == 0) {
        throw std::runtime_error("Failed to write " + filename);
    }
}

void GPUImage::read_png(const std::string& filename) {
    int w, h, n;
    unsigned char* data = stbi_load(filename.c_str(), &w, &h, &n, 4);

    if(m_N != w || m_N != h) {
        // todo handle all this gracefully
        throw std::runtime_error("unable to load image. check world size and whether image is square");
    }
    if(n != 4) {
        throw std::runtime_error("unable to load image. number of components != 4");
    }

    memcpy(m_data, data, w * h * n);

    stbi_image_free(data);
}

}
