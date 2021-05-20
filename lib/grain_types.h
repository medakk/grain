#pragma once

#include <cstdint>

namespace grain {

// todo would be nice if this could be combined with `GrainType`
using grain_t = uint32_t;

class GrainType {
public:

    // a hack for now, use the rightmost bit to store whether this cell has been
    // updated or not
    static const grain_t MASK_TYPE = 0xfffffffe;
    static const grain_t MASK_TURN = 0x00000001;

    static const grain_t Undefined = 0xffff00ff & MASK_TYPE;
    static const grain_t Blank     = 0xff444444 & MASK_TYPE;
    static const grain_t Sand      = 0xffc4a75c & MASK_TYPE;
    static const grain_t Water     = 0xff20acf7 & MASK_TYPE;
    static const grain_t Lava      = 0xff9e2416 & MASK_TYPE;
    static const grain_t Smoke     = 0xff3f3b33 & MASK_TYPE;

    static const grain_t Debug0    = 0xff00ff00 & MASK_TYPE;
    static const grain_t Debug1    = 0xff00cc00 & MASK_TYPE;
    static const grain_t Debug2    = 0xff00aa00 & MASK_TYPE;
    static const grain_t Debug3    = 0xff005500 & MASK_TYPE;
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