#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "../model/data/network_state.hpp"

template <typename T> struct FwdData {
    NetworkState<T> &states;
    T *input;
};

#endif
