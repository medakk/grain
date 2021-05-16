#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    std::cout << "Hello from C++" << __cplusplus << "\n";

    int driverVersion = 0;
    int runtimeVersion = 0;
    cuDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Driver/Runtime version: " << driverVersion << "/" << runtimeVersion << "\n";

    return 0;
}
