#ifndef MODEL_DATA_PARAMETERS
#define MODEL_DATA_PARAMETERS
#include "../../tools/tensor/tensor.hpp"

template <typename T> struct Parameters {
    tensor::Tensor<T> weights;
    tensor::Tensor<T> biases;

    Parameters() = default;
    Parameters(tensor::dims_t const &weights_dims,
               tensor::dims_t const &biases_dims)
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
    tensor::Tensor<T> weights;
    tensor::Tensor<T> biases;
    tensor::Tensor<T> input;
};

#endif
