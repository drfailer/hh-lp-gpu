#ifndef MODEL_DATA_PARAMETERS
#define MODEL_DATA_PARAMETERS
#include "tensor.hpp"

template <typename T>
struct Parameter {
    Tensor<T> *weights = nullptr;
    Tensor<T> *biases = nullptr;
};

#endif
