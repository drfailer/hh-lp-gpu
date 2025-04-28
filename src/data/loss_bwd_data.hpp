#ifndef DATA_LOSS_BWD_DATA_H
#define DATA_LOSS_BWD_DATA_H
#include "layer_state.hpp"

template <typename T> struct LossBwdData {
    NetworkState<T> &states;
    T *input;
    T *ground_truth;
    T *error;
    T learning_rate;
};

#endif
