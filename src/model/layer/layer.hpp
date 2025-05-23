#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_state.hpp"
#include "../../model/data/network_state.hpp"
#include "../../model/data/shape.hpp"

template <typename T> struct Layer {
    Layer(dims_t dims, shape_t shape = {})
        : dims(dims), parameter_shape(shape) {}
    size_t idx = 0;
    dims_t dims;
    shape_t parameter_shape;

    void create_state(NetworkState<T> &states) const {
        states.layers[idx] = create_state();
    }

    Tensor<T> *fwd(NetworkState<T> &states, Tensor<T> *input) {
        return fwd(states.layers[idx], input);
    }

    Tensor<T> *bwd(NetworkState<T> &states, Tensor<T> *error) {
        return bwd(states.layers[idx], error);
    }

    virtual void init(NetworkState<T> &state, int64_t batch_count) {
        dims.batch_count = batch_count;
        if (parameter_shape.dims.weights.size() > 0) {
            parameter_shape.dims.weights[0] = batch_count;
        }
        if (parameter_shape.dims.biases.size() > 0) {
            parameter_shape.dims.biases[0] = batch_count;
        }
        init(state.layers[idx], batch_count);
        // maybe we should init the optimizer functor here
    }

    virtual layer_state_t<T> create_state() const = 0;
    virtual void init(layer_state_t<T> &state, int64_t batch_count) = 0;
    virtual Tensor<T> *fwd(layer_state_t<T> &states, Tensor<T> *input) = 0;
    virtual Tensor<T> *bwd(layer_state_t<T> &states, Tensor<T> *error) = 0;
};

#endif
