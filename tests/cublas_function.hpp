#ifndef CUDNN_TESTS_CUBLAS_FUNCTION
#define CUDNN_TESTS_CUBLAS_FUNCTION
#include "../src/tools/gpu.hpp"
#include "utest.hpp"
#include <cublas_v2.h>
#include <cudnn.h>

extern cudnnHandle_t CUDNN_HANDLE;
extern cublasHandle_t CUBLAS_HANDLE;

UTest(matvecmul_n);
UTest(matvecmul_t);
UTest(matvecmul_batch_n);
UTest(matmul_n_n);
UTest(matmul_t_n);
UTest(matmul_n_t);
UTest(matmul_t_t);
UTest(matmul_batch_n_n);

#endif
