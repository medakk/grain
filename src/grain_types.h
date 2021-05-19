#pragma once

#include <cstdint>

namespace grain {

class GrainType {
public:

    static const uint32_t MASK_TYPE = 0xfffffffe;
    static const uint32_t MASK_TURN = 0x00000001;

    static const uint32_t Undefined = 0xffff00ff & MASK_TYPE;
    static const uint32_t Blank     = 0xff444444 & MASK_TYPE;
    static const uint32_t Sand      = 0xffc4a75c & MASK_TYPE;

    static const uint32_t Debug0    = 0xff00ff00 & MASK_TYPE;
    static const uint32_t Debug1    = 0xff00cc00 & MASK_TYPE;
    static const uint32_t Debug2    = 0xff00aa00 & MASK_TYPE;
    static const uint32_t Debug3    = 0xff005500 & MASK_TYPE;
};

struct EventData {
    bool reset {false};
    bool mouse_pressed {false};
    bool paused {false};
    bool screenshot {false};
    bool next_brush {false};
    bool prev_brush {false};

    float mouse_x, mouse_y;
};

}