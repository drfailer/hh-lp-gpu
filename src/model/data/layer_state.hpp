#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "parameters.hpp"

template <typename T> struct LayerState {
    tensor::Tensor<T> const *input = nullptr; // input of the forward pass (gpu)
    tensor::Tensor<T> output;                 // output of the forward pass (gpu)
    Parameters<T> parameters;
    Gradients<T> gradients;

    LayerState() = default;
    LayerState(Parameters<T> &&parameters) {
        this->parameters = std::move(parameters);
    }

    void create_gradient_tensors() {
        if (!this->parameters.weights.empty()) {
            this->gradients.weights.reshape_like(this->parameters.weights);
        }
        if (!this->parameters.biases.empty()) {
            this->gradients.biases.reshape_like(this->parameters.biases);
        }
    }
};

#endif
