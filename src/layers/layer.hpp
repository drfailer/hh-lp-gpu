#ifndef LAYERS_LAYER_H
#define LAYERS_LAYER_H
#include "../data/layer_state.hpp"

template <typename T>
struct Layer {
    Layer(auto dims): dims(dims) {}
    size_t idx = 0;
    LayerDimentions dims;

    void init(NetworkState<T> &states) {
        init(states.layer_states[idx]);
    }

    T *fwd(NetworkState<T> &states, T *input) {
        return fwd(states.layer_states[idx], input);
    }

    T *bwd(NetworkState<T> &states, T *error) {
        return bwd(states.layer_states[idx], error);
    }

    virtual void init(LayerState<T> &state) = 0;
    virtual T *fwd(LayerState<T> &states, T *input) = 0;
    virtual T *bwd(LayerState<T> &states, T *error) = 0;
};

#endif
