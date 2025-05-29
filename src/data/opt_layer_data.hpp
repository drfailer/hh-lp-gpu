#ifndef DATA_OPT_LAYER_DATA_H
#define DATA_OPT_LAYER_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct OptLayerData {
    std::shared_ptr<NNState<T>> state;
    size_t idx;
};

#endif
