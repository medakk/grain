#pragma once

#include "grain_types.h"
#include <MiniFB.h>

namespace grain {

class MiniFBRenderer {
public:
    template<typename F>
    static void start(F compute_buffer_func, EventData& event_data,
                      size_t width, size_t height) {
        // todo who frees this?
        mfb_window *window = mfb_open_ex("grain", 720, 720, WF_RESIZABLE);
        if(!window) {
            throw std::runtime_error("Unable to open window");
        }
        mfb_set_keyboard_callback(window, keyboard_callback);
        mfb_set_mouse_button_callback(window, mouse_button_callback);

        mfb_set_user_data(window, &event_data);

        do {
            // figure out mouse location
            const int mouse_x = mfb_get_mouse_x(window);
            const int mouse_y = mfb_get_mouse_y(window);
            event_data.mouse_x = (float)mouse_x / (float)mfb_get_window_width(window);
            event_data.mouse_y = (float)mouse_y / (float)mfb_get_window_height(window);
            event_data.mouse_x = std::clamp(event_data.mouse_x, 0.0f, 1.0f);
            event_data.mouse_y = std::clamp(event_data.mouse_y, 0.0f, 1.0f);

            const uint32_t *buffer = compute_buffer_func();

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
    static void keyboard_callback(mfb_window *window, mfb_key key,
                                         mfb_key_mod mod, bool isPressed) {
        if (key == KB_KEY_ESCAPE) {
            mfb_close(window);
        }
        EventData& event_data = *((EventData*) mfb_get_user_data(window));
        if(key == KB_KEY_R && isPressed) {
            event_data.should_reset = true;
        }
        if(key == KB_KEY_SPACE && isPressed) {
            event_data.is_paused = !event_data.is_paused;
        }
    }

    static void mouse_button_callback(struct mfb_window *window, mfb_mouse_button button,
                                      mfb_key_mod mod, bool isPressed) {
        if (button != MOUSE_BTN_1) {
            return;
        }

        EventData& event_data = *((EventData*) mfb_get_user_data(window));
        event_data.is_mouse_pressed = isPressed;
    }
};
}