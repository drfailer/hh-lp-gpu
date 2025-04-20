#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H
#include "log.h/log.h"
#include <cublas_v2.h>
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

template <typename T>
auto matvecmul(cublasHandle_t handle, bool trans, size_t rows, size_t cols,
               T *mat, T *vec, T *out) {
    cublasOperation_t cublas_trans =
        trans ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
    size_t ldmat = trans ? rows : cols;
    T alpha = 1, beta = 1;

    INFO_GRP("gemv: C = op(A) * X + Y", INFO_GRP_CUBLAS)
    INFO_GRP("op(A)[" << rows << ", " << cols << "] = " << mat, INFO_GRP_CUBLAS);
    INFO_GRP("X[" << cols << "] = " << vec, INFO_GRP_CUBLAS);
    INFO_GRP("C[" << rows << "] = " << out, INFO_GRP_CUBLAS);

    // we will only use float in this program, but there is still the
    // possibility to ad support for more
    return cublasSgemv_v2(handle, cublas_trans, cols, rows, &alpha, mat, ldmat,
                          vec, 1, &beta, out, 1);
}

template <typename T>
auto matmul(cublasHandle_t handle, bool A_trans, bool B_trans, size_t m,
            size_t n, size_t k, T const *A, T const *B, T *C) {
    cublasOperation_t cublas_trans_A = A_trans ? cublasOperation_t::CUBLAS_OP_N
                                               : cublasOperation_t::CUBLAS_OP_T;
    cublasOperation_t cublas_trans_B = B_trans ? cublasOperation_t::CUBLAS_OP_N
                                               : cublasOperation_t::CUBLAS_OP_T;
    size_t lda = A_trans ? m : k;
    size_t ldb = B_trans ? k : n;
    size_t ldc = n;
    T alpha = 1, beta = 0;

    INFO_GRP("gemm: C = op(A) * op(B) + C", INFO_GRP_CUBLAS)
    INFO_GRP("op(A)[" << m << ", " << k << "] = " << A, INFO_GRP_CUBLAS);
    INFO_GRP("op(B)[" << k << ", " << n << "] = " << B, INFO_GRP_CUBLAS);
    INFO_GRP("C[" << m << ", " << n << "] = " << C, INFO_GRP_CUBLAS);

    return cublasSgemm_v2(handle, cublas_trans_A, cublas_trans_B, n, m, k,
                          &alpha, B, ldb, A, lda, &beta, C, ldc);
}

#endif
