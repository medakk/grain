#pragma once

#include <cstdint>

namespace grain {
class GrainType {
public:
    static const uint32_t Undefined = 0xffff00ff;
    static const uint32_t Blank = 0xff000000;
    static const uint32_t Sand = 0xffc4a75c;
};
}