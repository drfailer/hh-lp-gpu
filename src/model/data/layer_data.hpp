#ifndef MODEL_DATA_LAYER_DATA
#define MODEL_DATA_LAYER_DATA
#include "../../tools/tensor/tensor.hpp"

template <typename T> struct LayerData {
    tensor::Tensor<T> x;
    tensor::Tensor<T> y;
    tensor::Tensor<T> dy;
    tensor::Tensor<T> dx;
    tensor::Tensor<T> w;
    tensor::Tensor<T> b;
    tensor::Tensor<T> dw;
    tensor::Tensor<T> db;
};

#endif
