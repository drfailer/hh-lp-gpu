#ifndef DATA_UPDATE_DATA_H
#define DATA_UPDATE_DATA_H
#include "layer_state.hpp"

template <typename T> struct UpdateData {
    NetworkState<T> &states;
    T learning_rate;
};

#endif
