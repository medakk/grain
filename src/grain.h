#pragma once

namespace grain{
class Grain {
public:
    static void step(const int *in, int *out, size_t n);
};
}