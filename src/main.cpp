#include <cuda_runtime.h>
#include "cxxopts.hpp"

#include "renderer.h"
#include "grain_types.h"
#include "grain_sim.h"

struct Options {
    size_t N{256};
    size_t speed{1};
    bool start_paused{false};
};

Options parse_args(int argc, char *argv[]) {
    cxxopts::Options options("grain", "GPU accelerated falling sand simulation");
    options.add_options()
            ("n,world-size", "size of world",
                    cxxopts::value<int>()->default_value("256"))
            ("s,speed", "number of iterations to run per update",
             cxxopts::value<int>()->default_value("1"))
            ("p,start-paused", "start with simulation paused. <space> to resume",
                    cxxopts::value<bool>()->default_value("false"))
            ("h,help", "print usage");
    const auto result = options.parse(argc, argv);

    if(result.count("help")) {
        std::cerr << options.help() << "\n";
        exit(0);
    }

    Options ret;
    ret.N = result["n"].as<int>();
    ret.speed = result["speed"].as<int>();
    ret.start_paused = result["start-paused"].as<bool>();

    return ret;
}

int main(int argc, char *argv[]) {
    Options options{};
    try {
        options = parse_args(argc, argv);
    } catch(std::exception& e) {
        std::cerr << "Failed to parse args: " << e.what() << "\n"
                  << "Run " << argv[0] << " -h for usage\n";
        exit(1);
    }

    grain::GrainSim grain_sim(options.N, options.speed);
    grain::EventData event_data;
    event_data.paused = options.start_paused;

    // create renderer and start update loop
    grain::MiniFBRenderer::start([&]() {
        const uint32_t* data = grain_sim.update(event_data);
        return data;
    }, event_data, options.N, options.N);

    return 0;
}
