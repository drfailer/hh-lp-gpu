#ifndef MODEL_DATA_LAYER_DATA
#define MODEL_DATA_LAYER_DATA
#include "../../tools/tensor/tensors.hpp"

template <typename T> struct LayerData {
    tensor::TensorView<T> x;
    tensor::Tensor<T> y;
    tensor::TensorView<T> dy;
    tensor::Tensor<T> dx;
    tensor::Tensor<T> w;
    tensor::Tensor<T> b;
    tensor::Tensor<T> dw;
    tensor::Tensor<T> db;
};

struct LayerParametersShape {
    tensor::TensorShape w = {};
    tensor::TensorShape b = {};
};

struct LayerIOShape {
    tensor::TensorShape x = {};
    tensor::TensorShape y = {};
};

#endif
