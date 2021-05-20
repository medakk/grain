#pragma once

#include <cstdint>

namespace grain {

// todo would be nice if this could be combined with `GrainType`
using grain_t = uint8_t;

class GrainType {
public:

    // a hack for now, use the rightmost bit to store whether this cell has been
    // updated or not
    static const grain_t MASK_TYPE = 0x7f;
    static const grain_t MASK_TURN = 0x80;

    static const grain_t Undefined = 0xff & MASK_TYPE;
    static const grain_t Blank     = 0x44 & MASK_TYPE;
    static const grain_t Sand      = 0x5c & MASK_TYPE;
    static const grain_t Water     = 0xf7 & MASK_TYPE;
    static const grain_t Lava      = 0x16 & MASK_TYPE;
    static const grain_t Smoke     = 0x33 & MASK_TYPE;

    static const grain_t Debug0    = 0x05 & MASK_TYPE;
    static const grain_t Debug1    = 0x0a & MASK_TYPE;
    static const grain_t Debug2    = 0x0d & MASK_TYPE;
    static const grain_t Debug3    = 0x08 & MASK_TYPE;
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