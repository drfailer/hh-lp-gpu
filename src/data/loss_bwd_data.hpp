#ifndef DATA_LOSS_BWD_DATA_H
#define DATA_LOSS_BWD_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct LossBwdData {
    NetworkState<T> &states;
    Tensor<T> *input;
    Tensor<T> *ground_truth;
    Tensor<T> *error;
    T learning_rate;
};

#endif
