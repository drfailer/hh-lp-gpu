cmake_minimum_required(VERSION 3.22)

project(test-cudnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

enable_language(CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_compile_options(-Wall -Wextra -Wuninitialized -fsanitize=address -fno-omit-frame-pointer -pedantic -g)
add_compile_options(-Wall -Wextra -Wuninitialized -pedantic -g -O3)

add_executable(test-cudnn src/main.cpp)
target_link_directories(test-cudnn PUBLIC ~/Programming/usr/lib/)
target_include_directories(test-cudnn PUBLIC ~/Programming/usr/include/
    /usr/local/cuda-12.8/include/ lib/cudnn-frontend/include/)
target_link_libraries(test-cudnn cudnn)
