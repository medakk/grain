#include "grain.h"
#include <cassert>

namespace grain {
__global__ void gpu_step(const int* in, int* out, size_t n) {
}

// todo how to avoid the explicit need for these wrappers?
void GrainSim::step(const int* in, int *out, size_t n) {

}
}
