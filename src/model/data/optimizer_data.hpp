#ifndef MODEL_DATA_OPTIMIZER_DATA
#define MODEL_DATA_OPTIMIZER_DATA
#include "../../tools/tensor/tensor.hpp"

template <typename T>
struct OptimizerData {
    tensor::tensor::Tensor<T> w;
    tensor::tensor::Tensor<T> b;
    tensor::tensor::Tensor<T> dw;
    tensor::tensor::Tensor<T> db;
    struct {
        tensor::TensorList<T> data;
    } ctx;
};

#endif
