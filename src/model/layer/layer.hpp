#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/layer_state.hpp"
#include "../../model/data/network_state.hpp"

template <typename T> struct Layer {
    Layer(auto dims) : dims(dims) {}
    size_t idx = 0;
    LayerDims dims;

    void init(NetworkState<T> &states) { init(states.layers[idx]); }

    T *fwd(NetworkState<T> &states, T *input) {
        return fwd(states.layers[idx], input);
    }

    T *bwd(NetworkState<T> &states, T *error) {
        return bwd(states.layers[idx], error);
    }

    virtual void init(LayerState<T> &state) = 0;
    virtual T *fwd(LayerState<T> &states, T *input) = 0;
    virtual T *bwd(LayerState<T> &states, T *error) = 0;
};

#endif
