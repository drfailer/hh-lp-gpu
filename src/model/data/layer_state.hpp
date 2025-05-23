#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "layer_data.hpp"
#include "parameters.hpp"

template <typename T> struct layer_state_t {
    Tensor<T> *input = nullptr;  // input of the forward pass (gpu)
    Tensor<T> *output = nullptr; // output of the forward pass (gpu)
    Tensor<T> *error = nullptr;  // output of the backwards pass (gpu)
    Parameter<T> *parameters = nullptr;
    Parameter<T> *gradients = nullptr;
    layer_data_t *layer_data = nullptr; // data used by layers
};

template <typename T> void destroy_layer_state(layer_state_t<T> &state) {
    delete state.output;
    delete state.error;
    if (state.parameters) {
        delete state.parameters->weights;
        delete state.parameters->biases;
        delete state.parameters;
    }
    if (state.gradients) {
        delete state.gradients->weights;
        delete state.gradients->biases;
        delete state.gradients;
    }
    delete state.layer_data;
}

#endif
