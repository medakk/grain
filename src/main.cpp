#include <string>

#include "cxxopts.hpp"
#include "grain_types.h"
#include "grain_sim.h"
#include "renderer.h"

struct Options {
    size_t N{256};
    size_t speed{1};
    size_t resolution{800};
    bool start_paused{false};
    bool verbose{false};
    std::string init_filename{};
};

Options parse_args(int argc, char *argv[]) {
    cxxopts::Options options("grain_bin", "GPU accelerated falling sand simulation");
    options.add_options()
            ("n,world-size", "size of world",
             cxxopts::value<int>()->default_value("256"))
            ("r,resolution", "render resolution(square)",
             cxxopts::value<int>()->default_value("800"))
            ("s,speed", "number of iterations to run per update",
             cxxopts::value<int>()->default_value("1"))
            ("p,start-paused", "start with simulation paused. <space> to resume",
             cxxopts::value<bool>()->default_value("false"))
            ("i,init-filename", "load initial state from .PNG file",
             cxxopts::value<std::string>()->default_value(""))
            ("v,verbose", "log information",
             cxxopts::value<bool>()->default_value("false"))
            ("h,help", "print usage");
    const auto result = options.parse(argc, argv);

    if(result.count("help")) {
        std::cerr << options.help() << "\n";
        exit(0);
    }

    Options ret;
    ret.N = result["n"].as<int>();
    ret.resolution = result["resolution"].as<int>();
    ret.speed = result["speed"].as<int>();
    ret.start_paused = result["start-paused"].as<bool>();
    ret.init_filename = result["init-filename"].as<std::string>();
    ret.verbose = result["verbose"].as<bool>();

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

    grain::GrainSim grain_sim(options.N, options.speed, options.init_filename);
    grain_sim.set_verbose(options.verbose);

    grain::EventData event_data;
    event_data.paused = options.start_paused;

    grain::OpenGLRenderer renderer(options.resolution, options.resolution);

    // create renderer and start update loop
    renderer.start(grain_sim, event_data,
                   options.resolution, options.resolution,
                   options.verbose);

    return 0;
}
