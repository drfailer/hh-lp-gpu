#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/layer_state.hpp"
#include "../../model/data/network_state.hpp"
#include "../../model/data/shape.hpp"

template <typename T> struct Layer {
    Layer(dims_t dims, shape_t shape = {}) : dims(dims), parameter_shape(shape) {}
    size_t idx = 0;
    dims_t dims;
    shape_t parameter_shape;

    void create_state(NetworkState<T> &states) const {
        states.layers[idx] = create_state();
    }

    T *fwd(NetworkState<T> &states, T *input) {
        return fwd(states.layers[idx], input);
    }

    T *bwd(NetworkState<T> &states, T *error) {
        return bwd(states.layers[idx], error);
    }

    virtual void init(int64_t batch_count) {
        dims.batch_count = batch_count;
        if (parameter_shape.dims.weights.size() > 0) {
            parameter_shape.dims.weights[0] = batch_count;
        }
        if (parameter_shape.dims.biases.size() > 0) {
            parameter_shape.dims.biases[0] = batch_count;
        }
        init();
        // maybe we should init the optimizer functor here
    }

    virtual LayerState<T> create_state() const = 0;
    virtual void init() = 0;
    virtual T *fwd(LayerState<T> &states, T *input) = 0;
    virtual T *bwd(LayerState<T> &states, T *error) = 0;
};

#endif
