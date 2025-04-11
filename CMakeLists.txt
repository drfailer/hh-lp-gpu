cmake_minimum_required(VERSION 3.22)

project(test-cudnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/lib/hedgehog/cmake_modules")
set(Hedgehog_INCLUDE_DIR lib/hedgehog/)

find_package(CUDAToolkit)
find_package(Hedgehog REQUIRED)

# add_compile_options(-Wall -Wextra -Wuninitialized -fsanitize=address -fno-omit-frame-pointer -pedantic -g)
add_compile_options(-Wall -Wextra -Wuninitialized -pedantic -g)

add_executable(test-cudnn
    src/main.cpp
    src/cudnn_operations.cpp
)
target_link_directories(test-cudnn
    PUBLIC
    ~/Programming/usr/lib/
)
target_include_directories(test-cudnn
    PUBLIC
    lib
    ~/Programming/usr/include/
    lib/cudnn-frontend/include/
    lib/hedgehog/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(test-cudnn
    PRIVATE
    cudnn
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)
