#ifndef DATA_OPT_DATA_H
#define DATA_OPT_DATA_H
#include "layer_state.hpp"

template <typename T> struct OptData {
    NetworkState<T> &states;
};

#endif
