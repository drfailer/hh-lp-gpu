#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "parameters.hpp"

template <typename T> struct LayerState {
    Tensor<T> const *input; // input of the forward pass (gpu)
    Tensor<T> output; // output of the forward pass (gpu)
    Parameters<T> parameters;
    Gradients<T> gradients;

    LayerState() = default;

    void set_parameters(Parameters<T> &&new_parameters) {
        parameters.weights = std::move(new_parameters.weights);
        parameters.biases = std::move(new_parameters.biases);
        create_parameters_gradients();
    }

    void create_parameters_gradients() {
        if (parameters.weights.data()) {
            gradients.weights.reshape(parameters.weights.dims(),
                                      parameters.weights.strides());
        }
        if (parameters.biases.data()) {
            gradients.biases.reshape(parameters.biases.dims(),
                                     parameters.biases.strides());
        }
    }
};

#endif
