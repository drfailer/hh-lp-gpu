#ifndef DATA_OPT_DATA_H
#define DATA_OPT_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct OptData {
    NetworkState<T> &states;
    T learning_rate;
};

#endif
