#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H
#include "utest.hpp"
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>

#ifndef UTEST
#define CUDNN_CHECK(expr)                                                      \
    {                                                                          \
        auto result = expr;                                                    \
        if (result.is_bad()) {                                                 \
            std::cerr << "[CUDNN_ERROR]: " __FILE__ ":" << __LINE__ << ": "    \
                      << result.get_message() << std::endl;                    \
        }                                                                      \
    }

#define CUDA_CHECK(expr)                                                       \
    {                                                                          \
        auto result = expr;                                                    \
        if (result != cudaSuccess) {                                           \
            std::cerr << "[CUDA_ERROR]: " __FILE__ ":" << __LINE__ << ": "     \
                      << cudaGetErrorString(result) << std::endl;              \
        }                                                                      \
    }

#define CUBLAS_CHECK(expr)                                                     \
    {                                                                          \
        auto result = expr;                                                    \
        if (result != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "[CUBLAS_ERROR]: " __FILE__ ":" << __LINE__ << ": "   \
                      << cublasGetStatusName(result) << std::endl;             \
        }                                                                      \
    }
#else
#define CUDNN_CHECK(expr)                                                      \
    {                                                                          \
        auto result = expr;                                                    \
        if (result.is_bad()) {                                                 \
            upanic("cudnn_error :: " + result.get_message());                  \
        }                                                                      \
    }

#define CUDA_CHECK(expr)                                                       \
    {                                                                          \
        auto result = expr;                                                    \
        if (result != cudaSuccess) {                                           \
            upanic("cuda_error :: " +                                          \
                   std::string(cudaGetErrorString(result)));                   \
        }                                                                      \
    }

#define CUBLAS_CHECK(expr)                                                     \
    {                                                                          \
        auto result = expr;                                                    \
        if (result != CUBLAS_STATUS_SUCCESS) {                                 \
            upanic("cublas_error :: " +                                        \
                   std::string(cublasGetStatusName(result)));                  \
        }                                                                      \
    }
#endif

template <typename T>
auto memcpy_host_to_gpu(T *dest_gpu, T *src_host, size_t size) {
    return cudaMemcpy(dest_gpu, src_host, size * sizeof(T),
                      cudaMemcpyHostToDevice);
}

template <typename T>
auto memcpy_gpu_to_host(T *dest_host, T *src_gpu, size_t size) {
    return cudaMemcpy(dest_host, src_gpu, size * sizeof(T),
                      cudaMemcpyDeviceToHost);
}

template <typename T>
auto memcpy_gpu_to_gpu(T *dest_gpu, T *src_gpu, size_t size) {
    return cudaMemcpy(dest_gpu, src_gpu, size * sizeof(T),
                      cudaMemcpyDeviceToDevice);
}

template <typename T> auto alloc_gpu(T **dest, size_t size) {
    return cudaMalloc((void **)dest, size * sizeof(T));
}

#endif
