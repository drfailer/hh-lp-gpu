cmake_minimum_required(VERSION 3.22)

project(hh-lp-gpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/lib/hedgehog/cmake_modules")
set(Hedgehog_INCLUDE_DIR lib/hedgehog/)

find_package(CUDAToolkit)
find_package(Hedgehog REQUIRED)
find_package(CUDA QUIET)
find_package(NVTX QUIET)

# CUDA options
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

enable_language(CUDA)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS})

# add_compile_options(-Wall -Wextra -Wuninitialized -fsanitize=address -fno-omit-frame-pointer -pedantic -g)
# add_compile_options(-Wall -Wextra -Wuninitialized -pedantic -g)
# add_compile_options(-g)
add_compile_options(-O3)

add_executable(hh-lp-gpu
    tests/main.cpp
    tests/cublas_function.cpp
    tests/layers.cpp
    src/kernels/linear_layer_kernel.cu
)
target_link_directories(hh-lp-gpu
    PUBLIC
    ~/Programming/usr/lib/
    /usr/local/cuda/lib64/
)
target_include_directories(hh-lp-gpu
    PUBLIC
    lib
    ~/Programming/usr/include/
    lib/hedgehog/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(hh-lp-gpu
    PRIVATE
    cudnn
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)
target_compile_definitions(hh-lp-gpu PUBLIC HH_USE_CUDA)

include(CTest)

add_test(NAME hh-lp-gpu COMMAND hh-lp-gpu)
