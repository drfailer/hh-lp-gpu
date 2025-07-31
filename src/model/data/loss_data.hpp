#ifndef MODEL_DATA_LOSS_DATA
#define MODEL_DATA_LOSS_DATA
#include "../../tools/tensor/tensor.hpp"

template <typename T>
struct LossData {
    tensor::tensor::Tensor<T> y;
};

#endif
