#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "layer_data.hpp"
#include "parameters.hpp"

template <typename T> struct LayerState {
    Tensor<T> *input = nullptr;  // input of the forward pass (gpu)
    Tensor<T> *output = nullptr; // output of the forward pass (gpu)
    Tensor<T> *error = nullptr;  // output of the backwards pass (gpu)
    parameters_t<T> parameters;
    parameters_t<T> gradients;
    layer_data_t *layer_data = nullptr; // data used by layers

    ~LayerState() {
        delete output;
        delete error;
        delete parameters.weights;
        delete parameters.biases;
        delete gradients.weights;
        delete gradients.biases;
        delete layer_data;
    }

    void set_parameters(parameters_t<T> const &parameters) {
        if (parameters.weights || parameters.biases) {
            this->parameters = parameters;
            this->gradients.weights = create_tensor<T>(
                parameters.weights->dims(), parameters.weights->strides());
            this->gradients.biases = create_tensor<T>(
                parameters.biases->dims(), parameters.biases->strides());
        }
    }

    template <typename DataType> DataType *get_data() {
        return dynamic_cast<DataType>(layer_data);
    }
};

#endif
