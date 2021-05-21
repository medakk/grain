#include <benchmark/benchmark.h>
#include "grain_sim.h"
#include "grain_types.h"

//TODO more fine-grained benchmarks. World size, speed, different initialization configs, etc

static void BM_Scenario00(benchmark::State& state) {
    // todo should assert the file exists or something

    grain::GrainSim grain_sim(1024, 20, "../bench/init_00.png");
    grain::EventData event_data{};

    const bool verbose = false;
    for(auto _ : state) {
        for(int i=0; i<100; i++) {
            grain_sim.update(event_data, verbose);
        }
    }
}
BENCHMARK(BM_Scenario00)->Unit(benchmark::kMillisecond);

static void BM_HomogenousLava(benchmark::State& state) {
    // todo should assert the file exists or something

    grain::GrainSim grain_sim(1024, 20, "../bench/homogenous_lava.png");
    grain::EventData event_data{};

    const bool verbose = false;
    for(auto _ : state) {
        for(int i=0; i<100; i++) {
            grain_sim.update(event_data, verbose);
        }
    }
}
BENCHMARK(BM_HomogenousLava)->Unit(benchmark::kMillisecond);

static void BM_HomogenousBlank(benchmark::State& state) {
    // todo should assert the file exists or something

    grain::GrainSim grain_sim(1024, 20, "../bench/homogenous_blank.png");
    grain::EventData event_data{};

    const bool verbose = false;
    for(auto _ : state) {
        for(int i=0; i<100; i++) {
            grain_sim.update(event_data, verbose);
        }
    }
}
BENCHMARK(BM_HomogenousBlank)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
