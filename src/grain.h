#pragma once

#include <cstdio>

#define cuda_assert(ans) { cuda_assert_((ans), __FILE__, __LINE__); }
inline void cuda_assert_(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Grain {
public:
    static void step(const int *in, int *out, size_t n);
    static void test_image(unsigned int *buf, size_t n);
};