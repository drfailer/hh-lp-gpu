cmake_minimum_required(VERSION 3.22)

project(test-cudnn)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_compile_options(-Wall -Wextra -Wuninitialized -fsanitize=address -fno-omit-frame-pointer -pedantic -g)
add_compile_options(-Wall -Wextra -Wuninitialized -pedantic -g)

add_executable(test-cudnn
    src/main.cpp
)
target_link_directories(test-cudnn
    PUBLIC
    ~/Programming/usr/lib/
)
target_include_directories(test-cudnn
    PUBLIC
    ~/Programming/usr/include/
    lib/cudnn-frontend/include/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(test-cudnn
    PRIVATE
    cudnn
    CUDA::cudart
    CUDA::cuda_driver
)
