#ifndef DATA_LOSS_FWD_DATA_H
#define DATA_LOSS_FWD_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct LossFwdData {
    std::shared_ptr<NNState<T>> states;
    Tensor<T> *input;
    Tensor<T> *ground_truth;
    Tensor<T> *result;
};

#endif
