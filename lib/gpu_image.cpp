#include "gpu_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace grain {

template<> int GPUImage<uint8_t>::n_components() const { return 1; }
template<> int GPUImage<uint32_t>::n_components() const { return 4; }

template<typename T>
void GPUImage<T>::write_png(const std::string& filename) const {
    if(stbi_write_png(filename.c_str(), m_N, m_N, n_components(),
                      m_data, 0) == 0) {
        throw std::runtime_error("Failed to write " + filename);
    }
}

template<typename T>
void GPUImage<T>::read_png(const std::string& filename) {
    int w, h, n;
    unsigned char* data = stbi_load(filename.c_str(), &w, &h, &n, n_components());

    if(m_N != w || m_N != h) {
        // todo handle all this gracefully
        throw std::runtime_error("unable to load image. check world size and whether image is square");
    }
    if(n != n_components()) {
        throw std::runtime_error("unable to load image. invalid number of components");
    }

    memcpy(m_data, data, w * h * n);

    stbi_image_free(data);
}


}
