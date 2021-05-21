#pragma once

#include <cstdint>

namespace grain {

// todo would be nice if this could be combined with `GrainType`
using grain_t = uint8_t;

class GrainType {
public:

    // Byte layout:
    // x  x  x  x  x  x  x  x
    // |       |_____________|
    // |             |
    // |             |
    // |             |
    // |             --> type(sand/water/etc)
    // |
    // --> turn indicator(0/1)

    static const grain_t MASK_TYPE = 0x1f;
    static const grain_t MASK_TURN = 0x80;

    static const grain_t Undefined = 0x00;
    static const grain_t Blank     = 0x01;
    static const grain_t Sand      = 0x02;
    static const grain_t Water     = 0x03;
    static const grain_t Lava      = 0x04;
    static const grain_t Smoke     = 0x05;
    static const grain_t Stone     = 0x06;
    static const grain_t Debug0    = 0x07;
    static const grain_t Debug1    = 0x08;
    static const grain_t Debug2    = 0x09;
    static const grain_t Debug3    = 0x0a;

    static const size_t MAX_TYPES = 32;

    // this has to be defined in the correct order. would be nice to look into a constexpr
    // way to keep everything organized. or a macro
    static constexpr uint32_t Colors[MAX_TYPES] = {
            0xffff00ff, // Undefined
            0xffb3c6b7, // Blank
            0xffe4d08c, // Sand
            0xff0086cc, // Water
            0xfff55545, // Lava
            0xff765d59, // Smoke
            0xff333333, // Stone
            0xff00ff00, // Debug0
            0xff00cc00, // Debug1
            0xff00aa00, // Debug2
            0xff005500, // Debug3
    };
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