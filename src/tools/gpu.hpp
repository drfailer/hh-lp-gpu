#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>

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
