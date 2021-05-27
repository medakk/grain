# grain

A CUDA-accelerated falling sand sim.

![](demo.gif)

## Build

_These instructions have been tested on a machine with Ubuntu 18.04, gcc8, CUDA 11.2_

Install:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* OpenGL Libraries (required by GLFW and ImGUI to render)

```bash
git clone --recursive https://github.com/medakk/grain.git
cd grain
mkdir build
cd build

# Pass in path to nvcc
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
cmake --build .
```

## Run

To run a demo:

```bash
$ ./grain_bin -n 1024 -s 20 -r 800
R:     Reset
S:     Screenshot(overwrites screenshot.png in current dir)
Q/E:   Previous/Next Brush
Space: Toggle pause
Esc:   Close
```

To see a full list of options:
```bash
$ ./grain_bin -h
GPU accelerated falling sand simulation
Usage:
  grain_bin [OPTION...]

  -n, --world-size arg     size of world (default: 256)
  -r, --resolution arg     render resolution(square) (default: 800)
  -s, --speed arg          number of iterations to run per update (default:
                           1)
  -p, --start-paused       start with simulation paused. <space> to resume
  -i, --init-filename arg  load initial state from .PNG file (default: )
  -v, --verbose            log information
  -h, --help               print usage
```

## License

(c) Karthik Karanth 2021, MIT License

Please refer to individual licenses for dependencies
