#pragma once

#include <cstdint>

namespace grain{

class GrainType {
public:
    static const uint32_t BLANK = 0xff000000;
    static const uint32_t SAND  = 0xffc4a75c;
};

class GrainSim {
public:
    static void step(const int *in, int *out, size_t n);
};
}