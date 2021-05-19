#pragma once

#include <cstdint>

namespace grain {

class GrainType {
public:

    static const uint32_t MASK_TYPE = 0xfffffffe;
    static const uint32_t MASK_TURN = 0x00000001;

    static const uint32_t Undefined = 0xffff00ff & MASK_TYPE;
    static const uint32_t Blank     = 0xff000000 & MASK_TYPE;
    static const uint32_t Sand      = 0xffc4a75c & MASK_TYPE;
};

struct EventData {
    bool should_reset {false};
    bool is_mouse_pressed {false};
    bool is_paused {false};
    bool should_take_screenshot {false};

    float mouse_x, mouse_y;
};

}