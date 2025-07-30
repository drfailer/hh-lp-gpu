#ifndef MODEL_DATA_NN_STATE_H
#define MODEL_DATA_NN_STATE_H
#include "layer_data.hpp"
#include "loss_state.hpp"
#include <vector>

template <typename T>
struct NNState {
    std::vector<LayerData<T>> layers;
    LossState<T> loss;
};

#endif
