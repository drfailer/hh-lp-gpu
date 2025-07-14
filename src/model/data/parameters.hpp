#ifndef MODEL_DATA_PARAMETERS
#define MODEL_DATA_PARAMETERS
#include "tensor.hpp"

template <typename T> struct Parameters {
    Tensor<T> weights;
    Tensor<T> biases;

    Parameters() = default;
    Parameters(tensor_dims_t const &weights_dims,
               tensor_dims_t const &biases_dims)
        : weights(weights_dims), biases(biases_dims) {}
    Parameters(Parameters &&other)
        : weights(std::move(other.weights)), biases(std::move(other.biases)) {}

    Parameters<T> const &operator=(Parameters<T> &&other) {
        this->weights = std::move(other.weights);
        this->biases = std::move(other.biases);
        return *this;
    }
};

template <typename T> struct Gradients {
    Tensor<T> weights;
    Tensor<T> biases;
    Tensor<T> input;
};

#endif
