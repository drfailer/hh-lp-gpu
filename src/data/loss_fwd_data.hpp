#ifndef DATA_LOSS_FWD_DATA_H
#define DATA_LOSS_FWD_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct LossFwdData {
    NetworkState<T> &states;
    Tensor<T> *input;
    Tensor<T> *ground_truth;
    Tensor<T> *result;
};

#endif
