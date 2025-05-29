#ifndef MODEL_DATA_CUDA_DATA
#define MODEL_DATA_CUDA_DATA
#include <cudnn_graph.h>
#include <cublas_v2.h>

struct cuda_data_t {
    cudnnHandle_t cudnn_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;
};

#endif
