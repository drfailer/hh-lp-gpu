#ifndef DATA_LOSS_BWD_DATA_H
#define DATA_LOSS_BWD_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct LossBwdData {
    std::shared_ptr<NNState<T>> states;
    tensor::Tensor<T> const *input;
    tensor::Tensor<T> const *ground_truth;
    tensor::Tensor<T> *error;
};

#endif
