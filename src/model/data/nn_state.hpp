#ifndef MODEL_DATA_NN_STATE_H
#define MODEL_DATA_NN_STATE_H
#include "layer_state.hpp"
#include "loss_state.hpp"
#include <vector>

template <typename T>
struct NNState {
    std::vector<LayerState<T>> layers;
    LossState<T> loss;
};

#endif
