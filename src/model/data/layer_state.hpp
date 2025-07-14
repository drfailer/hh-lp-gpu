#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "parameters.hpp"

template <typename T> struct LayerState {
    Tensor<T> const *input = nullptr; // input of the forward pass (gpu)
    Tensor<T> output;                 // output of the forward pass (gpu)
    Parameters<T> parameters;
    Gradients<T> gradients;

    LayerState() = default;
    LayerState(Parameters<T> &&parameters) {
        this->parameters = std::move(parameters);
    }

    void create_gradient_tensors() {
        if (this->parameters.weights.data()) {
            this->gradients.weights.reshape_like(this->parameters.weights);
        }
        if (this->parameters.biases.data()) {
            this->gradients.biases.reshape_like(this->parameters.biases);
        }
    }
};

#endif
