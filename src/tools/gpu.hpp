#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H
#include "log.h/log.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <vector>

#define CUDNN_CHECK(expr)                                                      \
    {                                                                          \
        auto _result = expr;                                                   \
        if (_result != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "[CUDA_ERROR]: " __FILE__ ":" << __LINE__ << ": "     \
                      << cudnnGetErrorString(_result) << std::endl;            \
        }                                                                      \
    }

#define CUDA_CHECK(expr)                                                       \
    {                                                                          \
        auto _result = expr;                                                   \
        if (_result != cudaSuccess) {                                          \
            std::cerr << "[CUDA_ERROR]: " __FILE__ ":" << __LINE__ << ": "     \
                      << cudaGetErrorString(_result) << std::endl;             \
        }                                                                      \
    }

#define CUBLAS_CHECK(expr)                                                     \
    {                                                                          \
        auto _result = expr;                                                   \
        if (_result != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "[CUBLAS_ERROR]: " __FILE__ ":" << __LINE__ << ": "   \
                      << cublasGetStatusName(_result) << std::endl;            \
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
auto memset_random_uniform_gpu(T *dest, size_t size, T lower_bound,
                               T higher_bound, int32_t seed = 0) {
    std::vector<T> mem_host(size);
    std::mt19937 mt(seed); // should be static?
    std::uniform_real_distribution<T> dist(lower_bound, higher_bound);

    for (size_t i = 0; i < size; ++i) {
        mem_host[i] = dist(mt);
    }
    return memcpy_host_to_gpu(dest, mem_host.data(), size);
}

template <typename T> auto memset_gpu(T *dest, size_t size, T value) {
    return cudaMemset(dest, value, size * sizeof(T));
}

template <typename T>
auto matvecmul(cublasHandle_t handle, bool trans, size_t rows, size_t cols,
               T alpha, T *mat, T *vec, T beta, T *out) {
    cublasOperation_t cublas_trans =
        trans ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
    size_t ldmat = cols;
    int m = cols;
    int n = rows;

    INFO_GRP("gemv: Y = alph * op(A) * X + beta * Y", INFO_GRP_CUBLAS_OPS)
    INFO_GRP("op(A)[" << (trans ? cols : rows) << ", " << (trans ? rows : cols)
                      << "] = " << mat,
             INFO_GRP_CUBLAS_OPS);
    INFO_GRP("X[" << (trans ? rows : cols) << "] = " << vec,
             INFO_GRP_CUBLAS_OPS);
    INFO_GRP("Y[" << (trans ? cols : rows) << "] = " << out,
             INFO_GRP_CUBLAS_OPS);

    // we will only use float in this program, but there is still the
    return cublasSgemv_v2(handle, cublas_trans, m, n, &alpha, mat, ldmat, vec,
                          1, &beta, out, 1);
}

template <typename T>
auto matvecmul(cublasHandle_t handle, bool trans, size_t rows, size_t cols,
               T alpha, T const *const *Aarray, T const *const *xarray, T beta,
               T *const *yarray, size_t batch_size) {
    cublasOperation_t cublas_trans =
        trans ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
    size_t lda = cols;
    int m = cols;
    int n = rows;

    // we will only use float in this program, but there is still the
    return cublasSgemvBatched(handle, cublas_trans, m, n, &alpha, Aarray, lda,
                              xarray, 1, &beta, yarray, 1, batch_size);
}

template <typename T>
auto matmul(cublasHandle_t handle, bool A_trans, bool B_trans, size_t m,
            size_t n, size_t k, T alpha, T const *A, T const *B, T beta, T *C) {
    cublasOperation_t cublas_trans_A = B_trans ? cublasOperation_t::CUBLAS_OP_T
                                               : cublasOperation_t::CUBLAS_OP_N;
    cublasOperation_t cublas_trans_B = A_trans ? cublasOperation_t::CUBLAS_OP_T
                                               : cublasOperation_t::CUBLAS_OP_N;
    size_t lda = A_trans ? m : k;
    size_t ldb = B_trans ? k : n;
    size_t ldc = n;

    INFO_GRP("gemm: C = op(A) * op(B) + C", INFO_GRP_CUBLAS_OPS)
    INFO_GRP("op(A)[" << m << ", " << k << "] = " << A, INFO_GRP_CUBLAS_OPS);
    INFO_GRP("op(B)[" << k << ", " << n << "] = " << B, INFO_GRP_CUBLAS_OPS);
    INFO_GRP("C[" << m << ", " << n << "] = " << C, INFO_GRP_CUBLAS_OPS);

    return cublasSgemm_v2(handle, cublas_trans_A, cublas_trans_B, n, m, k,
                          &alpha, B, ldb, A, lda, &beta, C, ldc);
}

template <typename T>
auto matmul(cublasHandle_t handle, bool A_trans, bool B_trans, size_t m,
            size_t n, size_t k, T alpha, T const *const *A, T const *const *B,
            T beta, T *const *C, size_t batch_size) {
    cublasOperation_t cublas_trans_A = B_trans ? cublasOperation_t::CUBLAS_OP_T
                                               : cublasOperation_t::CUBLAS_OP_N;
    cublasOperation_t cublas_trans_B = A_trans ? cublasOperation_t::CUBLAS_OP_T
                                               : cublasOperation_t::CUBLAS_OP_N;
    size_t lda = A_trans ? m : k;
    size_t ldb = B_trans ? k : n;
    size_t ldc = n;

    return cublasSgemmBatched(handle, cublas_trans_A, cublas_trans_B, n, m, k,
                              &alpha, B, ldb, A, lda, &beta, C, ldc,
                              batch_size);
}

#endif
