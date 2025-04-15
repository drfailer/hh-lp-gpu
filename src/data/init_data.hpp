#ifndef DATA_INIT_DATA_H
#define DATA_INIT_DATA_H
#include "layer_state.hpp"

template <typename T> struct InitData {
    NetworkState<T> &states;
};

#endif
