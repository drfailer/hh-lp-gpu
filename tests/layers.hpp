#ifndef CUDNN_TESTS_LAYERS
#define CUDNN_TESTS_LAYERS
#include "../src/tools/gpu.hpp"
#include "utest.hpp"
#include <cublas_v2.h>
#include <cudnn.h>

extern cudnnHandle_t CUDNN_HANDLE;
extern cublasHandle_t CUBLAS_HANDLE;

UTest(linear_layer_fwd);
UTest(linear_layer_bwd);
UTest(linear_layer_fwd_batched);
UTest(linear_layer_bwd_batched);
UTest(sigmoid_activation_fwd);
UTest(sigmoid_activation_bwd);
UTest(sgd_optimizer);
UTest(inference);
UTest(training);
UTest(mnist);
UTest(mnist_batched);

#endif
