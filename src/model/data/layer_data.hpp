#ifndef MODEL_DATA_LAYER_DATA
#define MODEL_DATA_LAYER_DATA
#include "../../tools/tensor/tensors.hpp"

struct LayerData {
    tensor::TensorView x;
    tensor::Tensor y;
    tensor::TensorView dy;
    tensor::Tensor dx;
    tensor::Tensor w;
    tensor::Tensor b;
    tensor::Tensor dw;
    tensor::Tensor db;
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
