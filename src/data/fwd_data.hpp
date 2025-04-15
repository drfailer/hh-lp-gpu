#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "layer_state.hpp"
#include <vector>

template <typename T> struct FwdData {
    NetworkState<T> &states;
    T *input;
};

#endif
