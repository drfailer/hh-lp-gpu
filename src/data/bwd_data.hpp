#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "layer_state.hpp"
#include <vector>

template <typename T> struct BwdData {
    NetworkState<T> &states;
    T *error;
};

#endif
