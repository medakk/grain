#pragma once

#include <MiniFB.h>

namespace grain {
class MiniFBRenderer {
public:
    MiniFBRenderer(size_t width_, size_t height_) :
            m_width(width_), m_height(height_) {

    }

    template<typename F>
    void start(F compute_buffer_func) {
        // todo who frees this?
        mfb_window *window = mfb_open_ex("my display", m_width, m_height, WF_RESIZABLE);
        if(!window) {
            throw std::runtime_error("Unable to open window");
        }

        do {
            const uint32_t *buffer = compute_buffer_func();
            const int state = mfb_update_ex(window, (void *) buffer, m_width, m_height);

            if (state < 0) {
                window = NULL;
                break;
            }
        } while(mfb_wait_sync(window));
    }

    //////////////////////////////////
    // Disable copying and assignment
    MiniFBRenderer(const MiniFBRenderer &) = delete;
    MiniFBRenderer(const MiniFBRenderer &&) = delete;
    MiniFBRenderer &operator=(const MiniFBRenderer &) = delete;
    MiniFBRenderer &operator=(const MiniFBRenderer &&) = delete;

private:
    size_t m_width, m_height;
};
}