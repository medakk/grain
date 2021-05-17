#pragma once

#include <MiniFB.h>

namespace grain {
class MiniFBRenderer {
public:
    template<typename F>
    static void start(F compute_buffer_func, size_t width, size_t height) {
        // todo who frees this?
        mfb_window *window = mfb_open_ex("grain", 720, 720, WF_RESIZABLE);
        if(!window) {
            throw std::runtime_error("Unable to open window");
        }
        mfb_set_keyboard_callback(window, minifb_keyboard_callback);

        size_t frame = 0;
        bool shouldReset = false;
        mfb_set_user_data(window, &shouldReset);

        do {
            const uint32_t *buffer = compute_buffer_func(frame++, shouldReset);
            if(shouldReset) {
                shouldReset = false;
            }
            const int state = mfb_update_ex(window, (void *) buffer, width, height);
            if (state < 0) {
                window = NULL;
                break;
            }
        } while(mfb_wait_sync(window));
    }

    //////////////////////////////////
    // Disable copying and assignment
    MiniFBRenderer(const MiniFBRenderer &) = delete;
    MiniFBRenderer &operator=(const MiniFBRenderer &) = delete;
    MiniFBRenderer(MiniFBRenderer &&) = delete;
    MiniFBRenderer &operator=(MiniFBRenderer &&) = delete;

private:
    static void minifb_keyboard_callback(mfb_window *window, mfb_key key,
                                         mfb_key_mod mod, bool isPressed) {
        if (key == KB_KEY_ESCAPE) {
            mfb_close(window);
        }
        if(key == KB_KEY_R && isPressed) {
            bool& shouldReset = *((bool*)mfb_get_user_data(window));
            shouldReset = true;
        }
    }
};
}