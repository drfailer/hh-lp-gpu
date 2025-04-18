#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "layer_state.hpp"

template <typename T> struct FwdData {
    NetworkState<T> &states;
    T *input;
};

#endif
