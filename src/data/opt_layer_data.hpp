#ifndef DATA_OPT_LAYER_DATA_H
#define DATA_OPT_LAYER_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct OptLayerData {
    NetworkState<T> &state;
    T learning_rate;
    size_t idx;
};

#endif
