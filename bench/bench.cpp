#include <benchmark/benchmark.h>
#include "grain_sim.h"
#include "grain_types.h"

//TODO more fine-grained benchmarks. World size, speed, different initialization configs, etc

static void BM_Step(benchmark::State& state) {
    grain::GrainSim grain_sim(1024, 20);
    grain::EventData event_data{};

    const bool verbose = false;
    for(auto _ : state) {
        for(int i=0; i<100; i++) {
            grain_sim.update(event_data, verbose);
        }
    }
}
BENCHMARK(BM_Step)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
