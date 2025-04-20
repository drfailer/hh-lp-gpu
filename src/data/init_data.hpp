#ifndef DATA_INIT_DATA_H
#define DATA_INIT_DATA_H
#include "layer_state.hpp"

enum class InitStatus {
    Init,
    Done,
};

template <typename T, InitStatus status = InitStatus::Init> struct InitData {
    NetworkState<T> &states;
};

#endif
