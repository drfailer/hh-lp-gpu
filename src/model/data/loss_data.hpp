#ifndef MODEL_DATA_LOSS_STATE
#define MODEL_DATA_LOSS_STATE
#include "../../tools/tensor/tensors.hpp"

template <typename T> struct LossData {
    tensor::Tensor tensor;
};

#endif
