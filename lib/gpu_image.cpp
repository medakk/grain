#include "gpu_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace grain {
void GPUImage::write_png(const std::string& filename) const {
    if(stbi_write_png(filename.c_str(), m_N, m_N, 4,
                      m_image, 0) == 0) {
        throw std::runtime_error("Failed to write " + filename);
    }
}
}
