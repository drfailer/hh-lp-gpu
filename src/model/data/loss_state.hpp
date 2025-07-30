#ifndef MODEL_DATA_LOSS_STATE
#define MODEL_DATA_LOSS_STATE
#include "../../tools/tensor/tensor.hpp"

template <typename T> struct LossState {
    tensor::Tensor<T> tensor;
};

#endif
