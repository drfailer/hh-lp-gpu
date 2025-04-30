#ifndef MODEL_DATA_NETWORK_STATE_H
#define MODEL_DATA_NETWORK_STATE_H
#include "layer_state.hpp"
#include <vector>

template <typename T>
struct NetworkState {
    std::vector<LayerState<T>> layers;
    T *loss;
};

#endif
