#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct BwdData {
    NetworkState<T> &states;
    T *error;
    T learning_rate;
};

#endif
